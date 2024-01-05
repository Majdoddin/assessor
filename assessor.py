import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
import torch.optim as optim
import boolean
from contextlib import nullcontext
import random
import sys

#export PYTHONPATH="${PYTHONPATH}:path/to/nanoGPT"
import model
from model import GPT, GPTConfig

def is_polish_normal_form(sequence):
    """
    Checks if the given sequence is a Boolean Formula in Polish normal form.

    The Polish notation (prefix notation) for Boolean formulas requires that:
    - Each operator must come before its operands.
    - Binary operators 'And', 'Or' take exactly two operands.
    - Unary operator 'Not' takes exactly one operand.
    - 'X1', 'X2', ..., 'Xn' are considered as variables and operands.

    Args:
    sequence (list): A sequence of strings representing the formula.

    Returns:
    bool: True if the sequence is a Boolean Formula in Polish normal form, False otherwise.
    Example usage:
    sequence = ["And", "Or", "X1", "Not", "X2", "X3"]
    is_valid = is_polish_normal_form(sequence)
    is_valid
    """

    # Stack to hold the count of operands needed for each operator
    operand_stack = []

    # Process the sequence in reverse order
    for token in reversed(sequence):
        if token in {"and", "or"}:
            # Binary operators require two operands
            if len(operand_stack) >= 2:
                # Pop two operands and push one as result of the binary operation
                operand_stack.pop()
                operand_stack.pop()
                operand_stack.append(1)
            else:
                # Not enough operands for a binary operator
                return False
        elif token == "not":
            # Unary operator requires one operand
            if operand_stack:
                # Pop one operand and push one as result of the unary operation
                operand_stack.pop()
                operand_stack.append(1)
            else:
                # Not enough operands for a unary operator
                return False
        else:
            # Variables and literals count as operands
            operand_stack.append(1)

    # Valid formula in Polish notation should leave exactly one operand on the stack
    return len(operand_stack) == 1

def sample_sequence(model, length, start_token_id, temperature=1.0, top_p=None, vocab_size=10, no_single_var = False):
    if top_p is not None and (top_p <= 0 or top_p >= 1):
        raise ValueError(f"top_k must in (0, ]")
    detached_tokens = None
    while True:
        if detached_tokens != None: print(f"too long: {[itos[t.item()] for t in detached_tokens[0]]}")
        detached_tokens = torch.full((1, 1), start_token_id, dtype=torch.long, device=next(model.parameters()).device)
        generated_logits = []
        for idx in range(length - 1):
            logits, loss = model(detached_tokens, targets = None)  #(batch_num, vocab_size)
            logits = logits[0, -1, :].unsqueeze(0)
            next_token_logits = logits / temperature

            #if no_single_var, first generated token should not be a var
            sm = F.softmax(next_token_logits, dim=-1)
            #no start_token
            sm[:, start_token_id] = 0
            if (idx == 0 and no_single_var):
                sm[:, 4:] = 0 #consider this by backward

            if top_p is not None:
                sorted_prob, sorted_idx = torch.sort(sm.squeeze(), descending=True)
                cp = torch.cumsum(sorted_prob, dim=-1)  #cumulative_probs
                sorted_prob[1:][cp[:-1] >= top_p] = 0
                next_token = torch.multinomial(sorted_prob, num_samples=1)
                next_token = sorted_idx[next_token].unsqueeze(0)
            else:
                next_token = torch.multinomial(sm, num_samples=1)  #(batch_size, 1)

            detached_tokens = torch.cat((detached_tokens, next_token.detach()), dim = 1)
            generated_logits.append(logits[0])

            if is_polish_normal_form([itos[t.item()] for t in detached_tokens[0, 1:]]):
                return detached_tokens[0], torch.stack(generated_logits, dim= 0)

def build_tree(tokens):
    if not tokens:
        return None

    token = tokens.pop(0)
    if token in ['and', 'or', 'not']:
        node = [token]
        # For 'not', expect exactly one operand, for 'and'/'or', expect at least two.
        expected_operands = 1 if token == 'not' else 2
        while expected_operands > 0 and tokens:
            operand, tokens = build_tree(tokens)
            node.append(operand)
            expected_operands -= 1
        return node, tokens
    else:
        # If the token is a variable, return it as is.
        return token, tokens

def to_nested_list(token_list):
    tree, remaining_tokens = build_tree(list(token_list))
    if remaining_tokens:
        raise ValueError("Invalid input: Unused tokens remain after parsing.")
    return tree

def to_parenthesized_string(tree):
    if isinstance(tree, str):
        return tree

    operator = tree[0]
    if operator == 'not':
        # For 'not', apply it to the single operand.
        return f"not {to_parenthesized_string(tree[1])}"
    else:
        # Otherwise, join the operands with the operator, applying recursion.
        operands = [to_parenthesized_string(operand) for operand in tree[1:]]
        return '(' + f" {operator} ".join(operands) + ')'

def expression_depth(expr):
    """
    Calculate the depth of a Boolean expression.

    :param expr: An expression from the boolean algebra module.
    :return: The depth of the expression.
    """
    # Base case: if the expression is a symbol, its depth is 0
    if isinstance(expr, boolean.Symbol):
        return 0

    # If the expression is a NOT operation, its depth is 1 + depth of the inner expression
    if isinstance(expr, boolean.NOT):
        return 1 + expression_depth(expr.args[0])

    # If the expression is an AND or OR operation, calculate the depth of each argument
    if isinstance(expr, (boolean.AND, boolean.OR)):
        return 1 + max(expression_depth(arg) for arg in expr.args)

    # For any other type of expression, return 0 as a default (though this should not happen)
    return 0

itos = {0:"ST", 1:"and", 2:"or", 3:"not", 4:"x1", 5:"x2", 6:"x3", 7:"x4", 8:"x5", 9:"x6", 10:"x7", 11:"x8", 12:"x9", 13:"x10", 14:"x11", 15:"x12"}
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

# Parameters for inference
temperature = 1 #0.7  # near 0 makes more deterministic
top_p = 0.9 # Top-k filtering, should be less than the vocabulary size

lasti = start = 0
end = 10000000
batch_loss = None
pos_samples = []
neg_samples = []

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

logf = 'output.txt'

for i in range(start, end):
    if eval:
        model.eval()
    else:
        model.train()
    ctx = torch.no_grad() if eval else nullcontext()
    with ctx:
        tokens, tokens_logs = sample_sequence(model, block_size, start_token_id, temperature=temperature, top_p=top_p if eval else None, vocab_size=vocab_size, no_single_var=min_depth>=2)
        tokens = tokens[1:] #token ST was not generated
        tknst = [itos[t.item()] for t in tokens]

        varn1 = len(set(s for s in tknst if s not in ['and', 'or', 'not']))
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

        print(tknst)
        print(f"depth:{depth} var_num:{varn2} simpified: {exp1}")
        with open(logf, 'a') as f:
            f.write(f"{tknst}\n")
            f.write(f"depth:{depth} var_num:{varn2} simpified: {exp1}\n")

        if min_depth > 1 and (depth == min_depth - 1) and not sec_round:
            continue #to avoid a shock

        hard = depth >= min_depth

        t = []
        if hard:
            for j in range(0, len(tokens)):
                t.append(torch.zeros(vocab_size))
                t[-1][tokens[j]] = 1.
            w = depth - min_depth + 1
            total_w += w
        else:
            for j in range(0, len(tokens)):
                t.append(torch.ones(vocab_size) / (vocab_size -1))
                t[-1][tokens[j]] = 0.

        target = torch.stack(t, dim=0)

        loss = cross_entropy(input=tokens_logs, target=target)
        if hard:
            pos_samples.append(w * loss)
        else:
            neg_samples.append(loss)

        if eval:
            if (i - lasti) > eval_num:
                eval = False
                if len(pos_samples) >= len(neg_samples):
                    torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f'state-depth-{min_depth}-{"2" if sec_round else "1"}.pt')
                    if (sec_round):
                        min_depth += 1
                        sec_round = False
                    else:
                        sec_round = True
                    print(f"min_depth = {min_depth}, {'sec_round' if sec_round else 'first_round'}, i = {i}")
                    with open(logf, 'a') as f:
                        f.write(f"min_depth = {min_depth}, {'sec_round' if sec_round else 'first_round'}, i = {i}\n")
                lasti = i
                pos_samples = []
                neg_samples = []
                total_w = 0
        elif (total_w >= b_size):
            p_batch_loss = sum(pos_samples)
            n_batch_loss = sum(neg_samples if len(neg_samples) < total_w else random.sample(neg_samples, total_w))
            p_batch_loss = p_batch_loss * torch.max(torch.tensor(1.0), 1.2 * n_batch_loss / p_batch_loss).item() #why 1.2?
            batch_loss = p_batch_loss + n_batch_loss
            batch_loss = batch_loss / (2 * total_w)

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_loss = None
            lasti = i

            eval = len(pos_samples) >= len(neg_samples)

            pos_samples = []
            neg_samples = []
            total_w = 0


