import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
import torch.optim as optim
import simplify
import boolean
import random

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

def sample_sequence(model, length, start_token_id, temperature=1.0, top_k=None, etw = 1. , vocab_size=10):
    """
    Sample a sequence of tokens from the model.

    Parameters:
    - model: the trained Transformer model for inference.
    - length: the total length of the sequence to generate, including the start token.
    - start_token_id: the token ID used to start the sequence.
    - temperature: controls the randomness of the output. Lower temperature means less random.
    - top_k: the number of highest probability vocabulary tokens to keep for top-k filtering.
    - vocab_size: the size of the model's vocabulary.

    Returns:
    - generated: the generated sequence of token IDs.
    """
    if top_k is not None and (top_k <= 0 or top_k >= vocab_size):
        raise ValueError(f"top_k must be a positive integer less than the vocabulary size {vocab_size}")

    detached_tokens = torch.full((1, 1), start_token_id, dtype=torch.long, device=next(model.parameters()).device)
    generated_logits = []
    for _ in range(length - 1):
        logits = model(detached_tokens)  #(batch_num, vocab_size)
        next_token_logits = logits[:, :] / temperature

        top_k = None #feature disabled
        if top_k is not None:
            # next_token_logits = top_k_logits(next_token_logits, top_k)
            next_token_logits = torch.topk(next_token_logits, top_k, 1)[0]  #(batch_num, top_k)

        sm = F.softmax(next_token_logits, dim=-1)
        sm[:, end_token_id] *= etw

        next_token = torch.multinomial(sm, num_samples=1)  #(batch_size, 1)
        detached_tokens = torch.cat((detached_tokens, next_token.detach()), dim = 0)
        generated_logits.append(logits[0])

        if next_token[0, 0] in (start_token_id, end_token_id):
            break
    return detached_tokens, torch.stack(generated_logits, dim= 0)

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

token_texts = {0:"ST", 1:"ET",  2:"and", 3:"or", 4:"not", 5:"x1", 6:"x2", 7:"x3", 8:"x4", 9:"x5"}
vocab_size = len(token_texts)  # Vocabulary size of the model

d_model = 512  # The dimension of the embeddings
nhead = 16  # Number of heads in the multi-head attention mechanisms
num_layers = 8  # Number of decoder layers
max_len = 42  # The length of sequence to generate, including the start and end tokens

# # Instantiate the model with a specific number of layers
model = Transformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, max_len=max_len)

# Parameters for inference
start_token_id = 0  # Assuming 1 is the ID for the start token in your vocabulary
end_token_id = 1
temperature = 1 #0.7  # near 0 makes more deterministic
top_k = None # Top-k filtering, should be less than the vocabulary size

optimizer = optim.Adam(model.parameters(), lr=1e-9)
# warmup_steps = 2000
# scheduler_warmup = WarmupScheduler(optimizer, warmup_steps, 1e-7, (1e-5) /5)
checkpoint = torch.load('state-length-4053.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.train()

#model.eval()
etw = 1.0
neg_num = 0
invalid_num = 0
start = 4053#1
end = 50000
# last_average = 1
batch_loss = None
pos_samples = []
neg_samples = []
max_invalid_len = 11.8#7
min_valid_len = 5 #1
ppp=0
init_neg_drop = 0.05
initial_long = 0
pos_lens = [0 for i in range(max_len)]
lasti = 0
fname = 'output-15.txt'
# with open('filename.txt', 'w') as file:pass
for i in range(start, end):
    # if i % 500 == 0:
    #     max_invalid_len += 1

    # if i > 299 and (i % 100 == 0):
    #         temperature /= 1.03
    tokens, tokens_logs = sample_sequence(model, max_len, start_token_id, temperature, top_k, etw, vocab_size)
    tokens = tokens.squeeze()
    if tokens.shape[0] <= 2 or ((start_token_id in tokens[1:]) or (torch.sum(tokens == end_token_id).item() != 1)):
        valid = 0.
    elif (4, 4) in zip(tokens.view(-1), tokens.view(-1)[1:]):
        valid = 0.
    # elif i < 200:
    #     valid = 1. #if is_polish_normal_form([token_texts[t] for t in tokens]) else 0.
    else:
        valid = 1. if is_polish_normal_form([token_texts[t.item()] for t in tokens[1:-1]]) else 0.

    tknst = [token_texts[t.item()] for t in tokens[1:-1]]

    if (i > 1000 and valid < 0.5):
        invalid_num += 1

    t = []
    if valid:
        for j in range(1, len(tokens)):
            t.append(torch.zeros(vocab_size))
            t[-1][tokens[j]] = 1.
    else:
        for j in range(1, len(tokens)):
            t.append(torch.ones(vocab_size) / (vocab_size -1))
            t[-1][tokens[j]] = 0.

    target = torch.stack(t, dim=0)

    w = 1.
    if valid > 0.5:# and i > 200:
        # varn1 = len(set(s for s in tknst if s not in ['and', 'or', 'not']))
        # nl = simplify.to_nested_list(tknst) #nested list
        # nl = simplify.deMorgan(nl)
        # nl = simplify.to_parenthesized_string(nl)
        # algebra = boolean.BooleanAlgebra()
        # exp1 = algebra.parse(nl, simplify=False)
        # exp1 = exp1.simplify()
        # exp1 = exp1.simplify()
        # varn2 = len(exp1.symbols)
        # depth = simplify.expression_depth(exp1)
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
        pos_lens[len(tknst)] += 1
        if len(tknst)<min_valid_len:
            continue
        w = 1.5**(len(tknst)+1-min_valid_len)

#        if (i > 1000):

        # if depth <= 2:
        #     w *= 2.2**((depth + 1 - (i)/800))
        # elif depth >= 3:
        #          w *= 2.2**((depth - 1 - (i)/8000))

        # w *= max(1, 1.5**((varn2 - 1 - (i)/1500)))  #length and actual depth are punished/rewarded, no need to punish here.

        if min_valid_len == 1:
            if len(tknst) > 1:
                initial_long += 1
            else:
                if (len(pos_samples) - initial_long >= initial_long):
                    continue
        assert w != 0

        print(f"{i} {len(tknst)} " + " ".join(tknst))
        with open(fname, 'a') as f:
            f.write(f"{i} {len(tknst)} " + " ".join(tknst) + "\n")

    else:
        if len(tknst) > max_invalid_len:  #??based on loss?
            continue
        # print(f"{varn2} {depth} {exp1}")
    # elif i > 200 and valid < 0.5:
    #     print(f"{len(tknst)}", end=" ")

    loss = cross_entropy(input=tokens_logs, target=target)
    if (valid > 0.5):
        ppp += len(tknst)
        pos_samples.append(w * loss)
    else:
        #print(len(tknst))
        if (random.random()<init_neg_drop): # to save memmory
            neg_samples.append(loss)
    if (len(pos_samples)>=8 and len(neg_samples) >= 8):
        with open(fname, 'a') as f:
            s = "-------------------------------------\n" + \
                f"min_valid_len:{min_valid_len}  max_invalid_len:{max_invalid_len}  av-len:{ppp/len(pos_samples):.1f}  iters: {i - lasti}\n" + \
                ' '.join(f"{i}:{pos_lens[i]}" for i in range(len(pos_lens)) if pos_lens[i] != 0) + f'  sum:{sum(pos_lens[1:])}  sum/iters:{sum(pos_lens[1:])/(i - lasti):.3f}'\
                "\n-------------------------------------"
            f.write(s + '\n')
            print(s)
        pos_lens=[0 for i in range(max_len)]
        lasti = i

        if (min_valid_len < ppp//len(pos_samples)):
            min_valid_len += 1
        max_invalid_len = max(ppp/len(pos_samples) + 6, 7)
        batch_loss = sum(pos_samples)
        batch_loss = batch_loss + sum(random.sample(neg_samples, 8))
        batch_loss = batch_loss / (len(pos_samples) + 8)

        batch_loss.backward()
        optimizer.step()
        #scheduler_warmup.step()
        optimizer.zero_grad()

        batch_loss = None
        pos_samples = []
        neg_samples = []
        ppp=0
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'state-length-{i}.pt')
print (invalid_num/(end - start - 1000))
#sample
# model.eval()
# tokens_logs = sample_sequence(model, max_length, start_token_id, temperature, top_k, vocab_size)
# print([token_texts[t] for t in tokens_logs])


torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'state-length-1.pt')