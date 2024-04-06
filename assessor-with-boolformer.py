#Train a model against boolformer to generate ever harder formulas
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

import logging

logging.basicConfig(filename='training.log', level=logging.DEBUG, format='%(message)s', filemode='a')
logger = logging.getLogger(__name__)

def find_depth(node):
    if not node.children: 
        return 1
    else:
        max_depth = 0
        for child in node.children:
            child_depth = find_depth(child)
            max_depth = max(max_depth, child_depth)
        return max_depth + 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

grad_clip = 0.75 #1.0

#tokens
itos = {0:"ST", 1:"and", 2:"or", 3:"not", 4:"var", 5:"x1", 6:"x2", 7:"x3", 8:"x4", 9:"x5", 10:"x6", 11:"x7", 12:"x8", 13:"x9", 14:"x10"}
start_tkn = 0
var_tkn = 4
#one for padding token
vocab_size = len(itos)
block_size=200

model_args = dict(n_layer=8, n_head=16, n_embd=512, block_size=block_size,
                bias=False, vocab_size=vocab_size, dropout=0)    #dropout, for pretraining 0 is good, for finetuning try 0.1+
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

weight_decay = 1e-1
learning_rate = 6e-5
beta1 = 0.9
beta2 = 0.95

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
temperature = 1 #0.7  # near 0 makes more deterministic
top_p = 0.9 # Top-p filtering, should be less than the vocabulary size

batch_loss = None

min_opn = 1
batch_size = 32 #128 #32 #*2 (hard and easy samples)
gradient_accumulation_steps = 4

checkpoint = None;#'state-opn-3-1.pt' #'name_of_checkpoint.pt'
#uncomment to_test_a_checkpoint
# sec_round = True #True iff checkpoint ends wiht 2
# min_opn = 6 #first number in checkpoint
# eval = True
# start = 0
# end = eval_num = 10000

if checkpoint:
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

boolformer_noiseless = load_boolformer(mode='noiseless', device=device)
if device_type == 'cuda':
    boolformer_noiseless.env.params.cpu = False
boolformer_noiseless.eval()
max_len = 15  #< block_size - 1
algebra = boolean.BooleanAlgebra()
t = algebra.parse(u'True', simplify=False)
f = algebra.parse(u'False', simplify=False)
high_pos = 0
batch_idx = 0
total_pos_n = 0
x_len = []

while True:
    batch_idx += 1    
    tokenss, tknsts, inputs, outputs, targets, too_long = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size
    for j in range(batch_size):
        model.eval()
        with torch.no_grad():
            with ctx:
                #top_p deactivated in train mode to increae exploration
                tokenss[j], too_long[j] = sample_formula(model, max_len, itos, start_tkn, var_tkn, temperature=temperature, top_p=top_p if eval else None)
        tknsts[j] = [itos[t.item()] for t in tokenss[j][1:]]

        #parsing the tokens
        nl = to_nested_list(tknsts[j])
        nl = to_parenthesized_string(nl)
        exp1 = algebra.parse(nl, simplify=False)

        #compute the output for all possible inputs
        smbs = list(exp1.symbols) #unsorted
        inputs[j] = np.array(list(product([f, t], repeat=len(smbs))))
        outputs[j] = np.array([exp1.subs({smbs[k]:inputs[j][l][k] for k in range(len(smbs))}).simplify() for l in range(len(inputs[j]))])
        inputs[j] = np.where(inputs[j] == t, True, False)
        outputs[j] = np.where(outputs[j] == t, True, False)
    #smbs[i] is Xi for boolformer
    torch.cuda.empty_cache()
    boolformer_noiseless.eval()
    with torch.no_grad():
        with ctx:
            pred_trees, error_arr, complexity_arr = boolformer_noiseless.fit(inputs, outputs, verbose=True, beam_size=10, beam_type="search")
    torch.cuda.empty_cache()
    logger.info("\n".join(
        f"{tknsts[j]}\n" +
        ('boolformer failed\n' if error_arr[j] != 0.0 else f'op_n:{complexity_arr[j]} simplified: {pred_trees[j]}')
        for j in range(batch_size)))

    #keep the same number of negative samples, mask out the rest  
    pos = [1 if s >= min_opn else 0 for s in complexity_arr]
    pos_n = pos.count(1)

    if pos_n == 0:
            logger.info(f"Batch {batch_idx}, positive ratio: 0, max_len: {max_len}, min_opn: {min_opn}")
            continue

    max_x_len = max(t.size()[0] for t in tokenss)
    x_len += [t.size()[0] for id, t in enumerate(tokenss) if not too_long[id]]

    #fixme remove extra negative samples
    #remove the last tokens
    xb = pad_sequence([t[:-1] for t in tokenss], batch_first=True, padding_value=start_tkn)   #b*(max_x_len-1)
    
    #reward formulas with more operators (after simplificatoin)
    #all tokens are rewarded/punishd if hard/easy
    #for variable tokens, token var is also rewarded/punished
    w = torch.ones(batch_size)
    for l in range(batch_size):
        hard = complexity_arr[l] >= min_opn
        target = torch.full((max_x_len-1, vocab_size), 0. if hard else 1.)
        for j in range(len(tokenss[l]) -1):
                    #next = torch.full((vocab_size,), 0. if hard else 1.)
                    target[j][tokenss[l][j+1]] = 1. if hard else 0.
                    #next[tokens[j]] = 1. if hard else 0.
                    if tokenss[l][j+1] > var_tkn:
                        target[j][var_tkn] = 1. if hard else 0.
        targets[l] = target
        #balance pos/neg samples, reward pos samples with bigger operation numbers
        if hard:
            w[l] *= (complexity_arr[l] - min_opn +1) 
    targets = torch.stack(targets).to(device)
    w = w.unsqueeze(1).unsqueeze(2).to(device)

    #mask paddings
    mask = (xb != start_tkn).float()
    mask[:, 0] = 1.0
    if pos_n < (batch_size / 2):
        zero_idx = [i for i, x in enumerate(pos) if x == 0]
        masked_neg_idx = sorted((random.sample(zero_idx, len(zero_idx) - pos_n)))
        for idx in masked_neg_idx:
            mask[idx] *= 0.0

    model.train()
    with ctx:
        logits, _ = model(idx = xb) #b*(max_x_len-1)*vocab_size
        loss = binary_cross_entropy_with_logits(input=logits, target=targets, weight=w, reduction = 'none')
        
        loss *= mask.unsqueeze(-1)
        loss = loss.sum() / (mask.sum() * len(itos))
    loss.backward()
    total_pos_n += pos_n
    if batch_idx % gradient_accumulation_steps == 0:    
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        #two consequent batches high ratio of pos samples
        if total_pos_n >= (batch_size * gradient_accumulation_steps) / 2:
            high_pos += 1
        else:
            high_pos = 0 
        if high_pos == 2:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'state-opn-{min_opn}.pt')
            min_opn += 1
            high_pos = 0
            max_len = int(min(max(max(x_len) + 5, max_len), block_size - 1))        
        logger.info(f"Batch {batch_idx // gradient_accumulation_steps}, total pos ratio: {total_pos_n / (batch_size * gradient_accumulation_steps):.2f}, loss: {loss:.2f}, not_too_long ratio: {len(x_len)/(batch_size * gradient_accumulation_steps):.2f}, max_len: {max_len}, min_opn: {min_opn}")
        total_pos_n = 0
        x_len = []
    # else:
    #     logger.info(f"Batch {batch_idx}, positive ratio: {pos_n/batch_size:.2f}, not_too_long ratio: {len(x_len)/batch_size:.2f}, max_len: {max_len}, min_opn: {min_opn}")
    del loss, pred_trees, error_arr, complexity_arr, xb, logits, w, mask, tokenss, tknsts, inputs, outputs, targets, too_long
    torch.cuda.empty_cache()

xxx = 1