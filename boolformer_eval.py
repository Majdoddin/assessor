import numpy as np
import boolformer
#from boolformer import load_boolformer, get_data_pmlb, run_models, get_logical_circuit, generate_data
from boolformer import load_boolformer
from graphviz import Source
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
import torch.optim as optim
import boolean
from contextlib import nullcontext
import random
import sys
from assessor import sample_sequence, to_parenthesized_string, to_nested_list, expression_depth

#nanoGPT of Andrej Karpathy
#export PYTHONPATH="${PYTHONPATH}:path/to/nanoGPT"
import model
from model import GPT, GPTConfig
from itertools import product

def rename_variables(formula):
    # Extract variables and sort them by their first appearance
    variables = [item for item in formula if item.startswith('x_')]
    unique_variables = sorted(set(variables), key=variables.index)

    # Create a mapping from old variable names to new ones
    mapping = {var: f'X_{i}' for i, var in enumerate(unique_variables)}

    # Replace the variable names in the formula
    renamed_formula = [mapping.get(item, item) for item in formula]

    return renamed_formula

#The goal is to generate ever harder formulas for boolformer
boolformer_noiseless = load_boolformer(mode='noiseless')
boolformer_noiseless.eval()

#tokens
itos = {0:"ST", 1:"and", 2:"or", 3:"not", 4:"x_0", 5:"x_1", 6:"x_2", 7:"x_3", 8:"x_4", 9:"x_5", 10:"x_6", 11:"x_7", 12:"x_8", 13:"x_9", 14:"x_10", 15:"x_11"}
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
top_p = 0.9 # Top-k filtering, should be less than the vocabulary size

lasti = start = 0
end = 10000000
batch_loss = None
hard_samples = []
easy_samples = []

total_w = 0
min_depth = 4
b_size = 32 #*2

eval = False
eval_num = 200

checkpoint = None
sec_round = False   #True iff checkpoint ends wiht 2

#uncomment to_test_a_checkpoint
checkpoint = 'state-depth-6-2.pt'
sec_round = True #True iff checkpoint ends wiht 2
min_depth = 6 #first number in checkpoint
eval = True
start = 0
end = eval_num = 1000

#checkpoint = 'state-depth-6-2.pt' #'name_of_checkpoint.pt'
if checkpoint:
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

logf = 'boolformer.txt'   #output file

for i in range(start, end):
    if eval:
        model.eval()
    else:
        model.train()
    ctx = torch.no_grad() if eval else nullcontext()
    with ctx:
        #top_p deactivated in train mode to increae exploration
        tokens, tokens_logs = sample_sequence(model, block_size, itos, start_token_id, temperature=temperature, top_p=top_p if eval else None, vocab_size=vocab_size, no_single_var=min_depth>=2)
        tokens = tokens[1:] #token ST was not generated
        tknst = [itos[t.item()] for t in tokens]
        #formula simplfication
        #tknst = rename_variables(tknst) #be carefull
        nl = to_nested_list(tknst)
        nl = to_parenthesized_string(nl)
        algebra = boolean.BooleanAlgebra()
        exp1 = algebra.parse(nl, simplify=False)
        exp1 = exp1.literalize()
        exp1 = exp1.simplify()
        exp1 = exp1.simplify()
        exp1 = exp1.literalize()

        varn2 = len(exp1.symbols)
        depth = expression_depth(exp1)

        if (varn2 > 10):
            continue
        smbs = list(exp1.symbols) #random order on variables
        t = algebra.parse(u'True', simplify=False)
        f = algebra.parse(u'False', simplify=False)
        inputs = np.array(list(product([f, t], repeat=len(smbs))))  #all inputs combinations
        outputs = np.array([exp1.subs({smbs[k]:inputs[j][k] for k in range(len(smbs))}).simplify() for j in range(len(inputs))])

        inputs = np.where(inputs == t, True, False)
        outputs = np.where(outputs == t, True, False)
        pred_trees, error_arr, complexity_arr = boolformer_noiseless.fit([inputs], [outputs], verbose=False, beam_size=10, beam_type="search")

        # Convert to numpy array
        print(f"depth:{depth} var_num:{varn2} complexity: simpified: {exp1}")
        print(f"error:{error_arr[0]} boolformer: {pred_trees[0]}")
        with open(logf, 'a') as f:
            f.write(f"depth:{depth} var_num:{varn2} complexity: simpified: {exp1}\n")
            f.write(f"error:{error_arr[0]} boolformer: {pred_trees[0]}\n")


        # for pred_tree in pred_trees:
        #     dot_graph = pred_tree.graphviz()
        #     dot_graph.save('tree_graph.dot')
        #     dot_graph.render('tree_graph', format='png', view=True, cleanup=True)
        #     img = Image.open('tree_graph.png')
        #     img.show()
