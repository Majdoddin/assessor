import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
import torch.optim as optim
import simplify
import boolean
import random
import sys

nanogpt_path = '/home/ruhollah/ai/nanoGPT'
sys.path.append(nanogpt_path)
import model
from model import GPT, GPTConfig

def is_completable_to_polish_normal_form(sequence):
    operand_stack = []

    for token in reversed(sequence):
        if token in {"and", "or"}:
            if operand_stack:
                operand_stack.pop()
            if operand_stack:
                operand_stack.pop()
            operand_stack.append(1)
        elif token == "not":
            # Unary operator requires one operand
            if operand_stack:
                # Pop one operand and push one as result of the unary operation
                operand_stack.pop()
            operand_stack.append(1)
        else:
            # Variables and literals count as operands
            operand_stack.append(1)

    # Valid formula in Polish notation should leave exactly one operand on the stack
    return len(operand_stack) == 1

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

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(max_len, 1, d_model))
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, tgt):
        seq_len, batch_size = tgt.size()
        if self.max_len < seq_len:
            raise ValueError(f"max_len ({self.max_len}) must be equal to or larger than sequence length ({seq_len})")
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)
        tgt_emb = self.embedding(tgt)
        pos_enc = self.positional_encoding[:seq_len, :, :].repeat(1, batch_size, 1)
        tgt = tgt_emb + pos_enc
        for layer in self.decoder_layers:
            tgt = layer(tgt, tgt_mask=tgt_mask)
        output = self.output_layer(tgt[-1, :, :])
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=self.positional_encoding.device), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

def top_k_logits(logits, k):
    if k == 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(1)
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

def sample_sequence(model, length, start_token_id, temperature=1.0, top_k=None, vocab_size=10):
    if top_k is not None and (top_k <= 0 or top_k >= vocab_size):
        raise ValueError(f"top_k must be a positive integer less than the vocabulary size {vocab_size}")
    detached_tokens = None
    while True:
        if detached_tokens != None: print(f"too long: {[itos[t.item()] for t in detached_tokens[0]]}")
        detached_tokens = torch.full((1, 1), start_token_id, dtype=torch.long, device=next(model.parameters()).device)
        generated_logits = []
        for _ in range(length - 1):
            logits, loss = model(detached_tokens, targets = None)  #(batch_num, vocab_size)
            logits = logits[0, -1, :].unsqueeze(0)
            next_token_logits = logits / temperature

            top_k = None #feature disabled
            if top_k is not None:
                # next_token_logits = top_k_logits(next_token_logits, top_k)
                next_token_logits = torch.topk(next_token_logits, top_k, 1)[0]  #(batch_num, top_k)

            sm = F.softmax(next_token_logits, dim=-1)

            sm[0, start_token_id] = 0
            next_token = torch.multinomial(sm, num_samples=1)  #(batch_size, 1)
            detached_tokens = torch.cat((detached_tokens, next_token.detach()), dim = 1)
            generated_logits.append(logits[0])

            if is_polish_normal_form([itos[t.item()] for t in detached_tokens[0, 1:]]):
                return detached_tokens[0], torch.stack(generated_logits, dim= 0)

# Parameters for the model

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, start_lr, end_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_steps:
            return [base_lr for base_lr in self.base_lrs]
        lr = self.start_lr + (self.end_lr - self.start_lr) * (self.last_epoch / self.warmup_steps)
        return [lr for _ in self.base_lrs]

itos = {0:"ST", 1:"and", 2:"or", 3:"not", 4:"x1", 5:"x2", 6:"x3", 7:"x4", 8:"x5"}
start_token_id = 0  # Assuming 1 is the ID for the start token in your vocabulary
#end_token_id = 1
vocab_size = len(itos)  # Vocabulary size of the model

d_model = 512  # The dimension of the embeddings
nhead = 16  # Number of heads in the multi-head attention mechanisms
num_layers = 8  # Number of decoder layers
block_size = 200  # The length of sequence to generate, including the start and end tokens

init_from = 'scratch'
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model_args = dict(n_layer=8, n_head=16, n_embd=512, block_size=block_size,
                  bias=False, vocab_size=vocab_size, dropout=0)    #dropout, for pretraining 0 is good, for finetuning try 0.1+
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
#model = Transformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, max_len=max_len)

# Parameters for inference
temperature = 1 #0.7  # near 0 makes more deterministic
top_k = None # Top-k filtering, should be less than the vocabulary size

weight_decay = 1e-1
learning_rate = 1e-6 # max learning rate
beta1 = 0.9
beta2 = 0.95
device = 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
#optimizer = optim.Adam(model.parameters(), lr=1e-9)

# warmup_steps = 2000
# scheduler_warmup = WarmupScheduler(optimizer, warmup_steps, 1e-7, (1e-5) /5)
# checkpoint = torch.load('state-length-4394.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.train()

#model.eval()

neg_num = 0
invalid_num = 0
lasti = start = 0#1
end = 100000
# last_average = 1
batch_loss = None
pos_samples = []
neg_samples = []
max_invalid_len = 10#7
min_valid_len = 4 #1
ppp=0
init_neg_drop = 0.5
initial_long = 0
pos_lens = [0 for i in range(block_size)]

# model.eval()
# with torch.no_grad():
#     for i in range (1000):
#         tokens, _ = sample_sequence(model, block_size, start_token_id, temperature=temperature, top_k=None, vocab_size=vocab_size)
#         print(tokens)

fname = 'output-15.txt'
# with open('filename.txt', 'w') as file:pass
total_w = 0
min_depth = 2
b_size = 16 #*2
for i in range(start, end):
    hard = False
    tokens, tokens_logs = sample_sequence(model, block_size, start_token_id, temperature=temperature, top_k=None, vocab_size=vocab_size)
    tokens = tokens[1:] #soc token was not generated
    tknst = [itos[t.item()] for t in tokens]

    varn1 = len(set(s for s in tknst if s not in ['and', 'or', 'not']))
    nl = simplify.to_nested_list(tknst) #nested list
    nl = simplify.deMorgan(nl)
    nl = simplify.to_parenthesized_string(nl)
    algebra = boolean.BooleanAlgebra()
    exp1 = algebra.parse(nl, simplify=False)
    exp1 = exp1.simplify()
    exp1 = exp1.simplify()
    varn2 = len(exp1.symbols)
    depth = simplify.expression_depth(exp1)
    hard = depth >= min_depth
    print(tknst)
    print(f"{depth} {exp1}")
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

    if hard:# and i > 200:
        pass
         #add demorgan again if needed
        # if (i > 1000 and varn2 == 0) or (i > 2000 and varn2 ==1):
        #     w = 0
        #w = (i - last_hit) * max(varn2 - ((i-200)/100)**0.4, 0)
        #w = 1.2**max(len(tknst) - ((i-200)/300), 1)

        # if ((i > 200) and min_valid_len == 1):
        #     min_valid_len = 2
        # if ((i > 500)and min_valid_len == 2):
        #     min_valid_len = 3
        # if ((i > 1200)and min_valid_len == 3):
        #     min_valid_len = 4
        # if ((i > 2400)and min_valid_len == 4):
        #     min_valid_len = 5
        # pos_lens[len(tknst)] += 1
        # if len(tknst)<min_valid_len:
        #     continue
        # w = (len(tknst)+1-min_valid_len)**0.3   #1.5**
#        if (i > 1000):

        # if depth <= 2:
        #     w *= 2.2**((depth + 1 - (i)/800))
        # elif depth >= 3:
        #          w *= 2.2**((depth - 1 - (i)/8000))

        # w *= max(1, 1.5**((varn2 - 1 - (i)/1500)))  #length and actual depth are punished/rewarded, no need to punish here.

        # if min_valid_len == 1:
        #     if len(tknst) > 1:
        #         initial_long += 1
        #     else:
        #         if (len(pos_samples) - initial_long >= initial_long):
        #             continue
        # assert w != 0

        # print(f"{i} {len(tknst)} " + " ".join(tknst))
        # with open(fname, 'a') as f:
        #     f.write(f"{i} {len(tknst)} " + " ".join(tknst) + "\n")
    else:
        pass
        # if len(tknst) > max_invalid_len:  #??based on loss?
        #     continue
        # print(f"{varn2} {depth} {exp1}")
    # elif i > 200 and valid < 0.5:
    #     print(f"{len(tknst)}", end=" ")

    loss = cross_entropy(input=tokens_logs, target=target)
    if hard:
        ppp += len(tknst)
        pos_samples.append(w * loss)
    else:
        #print(len(tknst))
        if (random.random()<init_neg_drop): # to save memmory
            neg_samples.append(loss)

#    if (len(pos_samples) >= b_size and len(neg_samples) >= b_size):
    if (total_w >= b_size and len(neg_samples) >= total_w):
        # with open(fname, 'a') as f:
            # s = "-------------------------------------\n" + \
            #     f"min_valid_len:{min_valid_len}  max_invalid_len:{max_invalid_len}  av-len:{ppp/len(pos_samples):.1f}  iters: {i - lasti}\n" + \
            #     ' '.join(f"{i}:{pos_lens[i]}" for i in range(len(pos_lens)) if pos_lens[i] != 0) + f'  sum:{sum(pos_lens[2:])}  sum/iters:{sum(pos_lens[1:])/(i - lasti):.3f}'\
            #     "\n-------------------------------------"
            # f.write(s + '\n')
            # print(s)
            # if (min_valid_len <= 4) or ((pos_lens[min_valid_len-1]+8) / (i - lasti)) >= 8/500:  #make sure it is better than random
            #     if (min_valid_len < ppp//len(pos_samples)):   #fixme: at least 4 have length min+2
            #         min_valid_len += 1
            #     if (max_invalid_len < ppp//len(pos_samples) + 6):
            #         max_invalid_len += 1
        #max_invalid_len = max(ppp/len(pos_samples) + 6, 7)
        p_batch_loss = sum(pos_samples)
        n_batch_loss = sum(random.sample(neg_samples, total_w))
        p_batch_loss = p_batch_loss * torch.max(torch.tensor(1.0), 1.2 * n_batch_loss / p_batch_loss).item() #why 1.2
        batch_loss = p_batch_loss + n_batch_loss
        batch_loss = batch_loss / (2 * total_w)

        batch_loss.backward()
        optimizer.step()
        #scheduler_warmup.step()
        optimizer.zero_grad()

        batch_loss = None

        # pos_lens=[0 for i in range(block_size)]
        lasti = i
        pos_samples = []
        neg_samples = []
        ppp=0
        total_w = 0
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     }, f'state-length-{i}.pt')
# print (invalid_num/(end - start - 1000))
#sample
# model.eval()
# tokens_logs = sample_sequence(model, max_length, start_token_id, temperature, top_k, vocab_size)
# print([token_texts[t] for t in tokens_logs])


torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'state-length-1.pt')