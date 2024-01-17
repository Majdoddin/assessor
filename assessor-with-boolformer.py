import torch
from torch.nn.functional import cross_entropy
import boolean
from contextlib import nullcontext
import random
import numpy as np
from boolformer import load_boolformer

#nanoGPT of Andrej Karpathy
#export PYTHONPATH="${PYTHONPATH}:path/to/nanoGPT"
import model
from model import GPT, GPTConfig
from itertools import product

from assessor import *

def find_depth(node):
    if not node.children:
        return 1
    else:
        max_depth = 0
        for child in node.children:
            child_depth = find_depth(child)
            max_depth = max(max_depth, child_depth)
        return max_depth + 1

#tokens
itos = {0:"ST", 1:"and", 2:"or", 3:"not", 4:"x1", 5:"x2", 6:"x3", 7:"x4", 8:"x5", 9:"x6", 10:"x7", 11:"x8", 12:"x9", 13:"x10"}
start_token_id = 0
vocab_size = len(itos)
block_size=200

model_args = dict(n_layer=8, n_head=16, n_embd=512, block_size=block_size,
                bias=False, vocab_size=vocab_size, dropout=0)    #dropout, for pretraining 0 is good, for finetuning try 0.1+
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

weight_decay = 1e-1
learning_rate = 1e-6
beta1 = 0.9
beta2 = 0.95
device = 'cpu' #current version tested only on cpu
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

temperature = 1 #0.7  # near 0 makes more deterministic
top_p = 0.9 # Top-p filtering, should be less than the vocabulary size

lasti = start = 29554
end = 10000000
batch_loss = None
hard_samples = []
easy_samples = []

total_w = 0
min_opn = 3
b_size = 32 #*2

eval = False
eval_num = 200

checkpoint = None
sec_round = False   #True iff checkpoint ends wiht 2

#uncomment to_test_a_checkpoint
# checkpoint = 'state-depth-6-2.pt'
# sec_round = True #True iff checkpoint ends wiht 2
# min_opn = 6 #first number in checkpoint
# eval = True
# start = 0
# end = eval_num = 10000

checkpoint = 'state-opn-3-29553.pt' #'name_of_checkpoint.pt'
if checkpoint:
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

logf = 'output20.txt'   #output file

#The goal is to generate ever harder formulas for boolformer
boolformer_noiseless = load_boolformer(mode='noiseless')
boolformer_noiseless.eval()

for i in range(start, end):
    if eval:
        model.eval()
    else:
        model.train()
    ctx = torch.no_grad() if eval else nullcontext()
    with ctx:
        #top_p deactivated in train mode to increae exploration
        tokens, tokens_logs = sample_sequence(model, block_size, itos, start_token_id, temperature=temperature, top_p=top_p if eval else None, vocab_size=vocab_size, no_single_var=min_opn>=2)
        tokens = tokens[1:] #token ST was not generated
        tknst = [itos[t.item()] for t in tokens]

        nl = to_nested_list(tknst)
        nl = to_parenthesized_string(nl)
        algebra = boolean.BooleanAlgebra()
        exp1 = algebra.parse(nl, simplify=False)

        smbs = list(exp1.symbols) #random order on variables
        t = algebra.parse(u'True', simplify=False)
        f = algebra.parse(u'False', simplify=False)
        inputs = np.array(list(product([f, t], repeat=len(smbs))))  #all inputs combinations
        outputs = np.array([exp1.subs({smbs[k]:inputs[j][k] for k in range(len(smbs))}).simplify() for j in range(len(inputs))])

        inputs = np.where(inputs == t, True, False)
        outputs = np.where(outputs == t, True, False)
        with torch.no_grad():
            pred_trees, error_arr, complexity_arr = boolformer_noiseless.fit([inputs], [outputs], verbose=False, beam_size=10, beam_type="search")
        opn = complexity_arr[0]  #number of ops in simplified formula

        s = str(tknst) + "\n"
        if error_arr[0] != 0.0:
            s += "boolformer failed\n"
        else:
            s += f"op_n:{opn} simpified: {pred_trees[0]} \n"
        print(s, end="")
        with open(logf, 'a') as f:
            f.write(s)

        if error_arr[0] != 0.0:  #fixme we can use it as positive sample
            continue

        #2 rounds of training for each depth. In the first round, simply ignore formulas with depth mindepth-1, to avoid a shock.
        if min_opn > 1 and (opn == min_opn - 1) and not sec_round:
            continue

        hard = opn >= min_opn

        t = []   #all tokens are rewarded/punishd if hard/easy
        if hard:
            for j in range(0, len(tokens)):
                t.append(torch.zeros(vocab_size))
                t[-1][tokens[j]] = 1.
            w = opn - min_opn + 1   #hard samples are weighted wrt their depth
            total_w += w
        else:
            for j in range(0, len(tokens)):
                t.append(torch.ones(vocab_size) / (vocab_size -1))
                t[-1][tokens[j]] = 0.

        target = torch.stack(t, dim=0)

        loss = cross_entropy(input=tokens_logs, target=target)
        if hard:
            hard_samples.append(w * loss)
        else:
            easy_samples.append(loss)

        if eval:   #no training in eval mode
            if (i - lasti) > eval_num:
                eval = False   #end of eval, back to training
                if len(hard_samples) >= len(easy_samples) / 2:  #high frequecy of hard samples in eval mode, increase the min_depth
                    torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f'state-opn-{min_opn}-{"2" if sec_round else "1"}.pt')
                    if (sec_round):
                        min_opn += 1
                        sec_round = False
                    else:
                        sec_round = True
                    s = f"min_opn = {min_opn}, {'sec_round' if sec_round else 'first_round'}, i = {i}"
                    print(s)
                    with open(logf, 'a') as f:
                        f.write(s + "\n")
                lasti = i
                hard_samples = []
                easy_samples = []
                total_w = 0
        elif (total_w >= b_size):   #a hard sample of weight w is counted w times
            p_batch_loss = sum(hard_samples)
            n_batch_loss = sum(easy_samples if len(easy_samples) < total_w else random.sample(easy_samples, total_w))
            p_batch_loss = p_batch_loss * torch.max(torch.tensor(1.0), 1.2 * n_batch_loss / p_batch_loss).item() #wanna hard samples to have a bit more weight
            batch_loss = p_batch_loss + n_batch_loss
            batch_loss = batch_loss / (2 * total_w)   #hard + easy samples

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_loss = None

            eval = len(hard_samples) >= len(easy_samples) / 2   #if half of samples are hard, run in eval mode (eventually increase min_depth)
            if eval:
                lasti = i
            elif i - lasti > 10000:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f'state-opn-{min_opn}-{i}.pt')
                lasti = i

            hard_samples = []
            easy_samples = []
            total_w = 0