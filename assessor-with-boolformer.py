import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
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
itos = {0:"ST", 1:"and", 2:"or", 3:"not", 4:"var", 5:"x1", 6:"x2", 7:"x3", 8:"x4", 9:"x5", 10:"x6", 11:"x7", 12:"x8", 13:"x9", 14:"x10"}
start_tkn = 0
var_tkn = 4
vocab_size = len(itos)
padding_token = -1
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

lasti = start = 4883
end = 10000000
batch_loss = None
hard_samples = []
easy_samples = []

total_w = 0
min_opn = 3
b_size = 32 #*2 (hard and easy samples)

eval = False
eval_num = 200

checkpoint = None
sec_round = True   #True iff checkpoint ends wiht 2

#uncomment to_test_a_checkpoint
# checkpoint = 'state-depth-6-2.pt'
# sec_round = True #True iff checkpoint ends wiht 2
# min_opn = 6 #first number in checkpoint
# eval = True
# start = 0
# end = eval_num = 10000

checkpoint = 'state-opn-3-1.pt' #'name_of_checkpoint.pt'
if checkpoint:
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

logf = 'output21.txt'   #output file

#The goal is to generate ever harder formulas for boolformer
boolformer_noiseless = load_boolformer(mode='noiseless')
boolformer_noiseless.eval()
max_len = 100  #< block_size - 1#30

samples = []

for i in range(start, end):
    if eval:
        model.eval()
    else:
        model.train()
    ctx = torch.no_grad() if eval else nullcontext()
    with ctx:
        #top_p deactivated in train mode to increae exploration
        with torch.no_grad():
            model.eval()
            tokens, _ = sample_formula(model, max_len, itos, start_tkn, var_tkn, temperature=temperature, top_p=top_p if eval else None, vocab_size=vocab_size, no_single_var=min_opn>=2)
        tokens = tokens[1:] #remove token ST
        tknst = [itos[t.item()] for t in tokens]

        algebra = boolean.BooleanAlgebra()

        #parsing the tokens
        nl = to_nested_list(tknst)
        nl = to_parenthesized_string(nl)
        exp1 = algebra.parse(nl, simplify=False)

        #compute the output for all possible inputs
        smbs = list(exp1.symbols) #unsorted
        t = algebra.parse(u'True', simplify=False)
        f = algebra.parse(u'False', simplify=False)
        inputs = np.array(list(product([f, t], repeat=len(smbs))))
        outputs = np.array([exp1.subs({smbs[k]:inputs[j][k] for k in range(len(smbs))}).simplify() for j in range(len(inputs))])
        inputs = np.where(inputs == t, True, False)
        outputs = np.where(outputs == t, True, False)

        #smbs[i] is Xi for boolformer
        with torch.no_grad():
            pred_trees, error_arr, complexity_arr = boolformer_noiseless.fit([inputs], [outputs], verbose=False, beam_size=10, beam_type="search")
        opn = complexity_arr[0]  #number of ops in simplified formula

        s = str(tknst) + "\n"
        if error_arr[0] != 0.0:
            s += "boolformer failed\n"
        else:
            s += f"op_n:{opn} simpilified: {pred_trees[0]} \n"
        print(s, end="")
        with open(logf, 'a') as f:
            f.write(s)

        if error_arr[0] != 0.0:  #TODO we can use it as positive sample
            continue

        #2 rounds of training for opn. In the first round, simply ignore formulas with depth mindepth-1, to avoid a shock.
        if min_opn > 1 and (opn == min_opn - 1) and not sec_round:
            continue

        hard = opn >= min_opn
        samples = []
        samples.append((tokens, opn))

        if eval:   #no training in eval mode
            if (i - lasti) > eval_num:
                eval = False   #end of eval, back to training
                if len(hard_samples) >= len(easy_samples) / 2:  #high frequecy of hard samples in eval mode, increase the min_depth
                    torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f'state-opn-{min_opn}-{"2" if sec_round else "1"}.pt')
                    if ((min_opn == 1) or sec_round):
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
            #todo you can shuffle samples, keep so many neg as pos samples.
            max_x_len = max([t.size()[0] for t, _ in samples])
            #fixme remove extra negative samples
            #remove the last tokens
            xb = pad_sequence([s[:-1] for s, _ in samples], batch_first=True, padding_value=padding_token)   #b*max_s_len
            logits, _ = model(xb)

            #reward formulas with more operators (after simplificatoin)
            #all tokens are rewarded/punishd if hard/easy
            #for variable tokens, token var is also rewarded/punished
            targets = []
            w = torch.ones(len(samples))
            for l, tokens, opn in enumerate(samples):
                hard = opn >= min_opn
                target = torch.full((max_x_len, vocab_size+1), 0. if hard else 1.)
                for j in range(0, len(tokens)):
                            #next = torch.full((vocab_size,), 0. if hard else 1.)
                            target[j][tokens[j]] = 1. if hard else 0.
                            #next[tokens[j]] = 1. if hard else 0.
                            if tokens[j] > var_tkn:
                                target[j][var_tkn] = 1. if hard else 0.
                targets.append(target)
                w[l] *= opn - min_opn + 1
            targets = torch.stack(targets)

            loss = binary_cross_entropy_with_logits(input=logits, target=target, weight=w)
            mask = (xb != padding_token).float()
            loss *= mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
            if hard:
                # for j in range(0, len(tokens)):
                #     next = torch.zeros(vocab_size)
                #     next[tokens[j]] = 1.
                #     if tokens[j] > var_tkn:
                #         next[var_tkn] = 1.
                #     t.append(next)
                w = opn - min_opn + 1   #hard samples are weighted wrt their depth
                total_w += w
                #By eval only count is important
                hard_samples.append(1 if eval else (w * loss))
            else:
                #To save memory. By eval only count is important
                easy_samples.append(1 if eval else (None if (len([elem for elem in easy_samples if elem is not None]) > 1.1 * total_w) else loss)) #all later easy samples are discarded. or should we do it randomly?


            eval = len(hard_samples) >= len(easy_samples) / 2   #if half of samples are hard, run in eval mode (eventually increase min_depth)
            easy_samples = [elem for elem in easy_samples if elem is not None]  #some

            p_batch_loss = sum(hard_samples)
            n_batch_loss = sum(easy_samples if len(easy_samples) < total_w else random.sample(easy_samples, total_w))
            p_batch_loss = p_batch_loss * torch.max(torch.tensor(1.0), 1.2 * n_batch_loss / p_batch_loss).item() #wanna hard samples to have a bit more weight
            batch_loss = p_batch_loss + n_batch_loss
            batch_loss = batch_loss / (2 * total_w)   #hard + easy samples

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_loss = None

            if eval:
                lasti = i
            elif i - lasti > 5000:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f'state-opn-{min_opn}-{i}.pt')
                lasti = i

            hard_samples = []
            easy_samples = []
            total_w = 0
xxx = 1
#9855