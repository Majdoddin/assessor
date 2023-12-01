import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.optim as optim
import simplify
import boolean

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
    generated_logits = torch.empty(0)
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
        generated_logits = torch.cat((generated_logits, logits[0][next_token][0]), dim=0)

        if next_token[0, 0] in (start_token_id, end_token_id):
            break
    return detached_tokens, generated_logits

# Parameters for the model

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

optimizer = optim.Adam(model.parameters(), lr=0.000001)
# checkpoint = torch.load('state-5000.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.train()

#model.eval()
etw = 1.0
last_hit = 200
invalid_num = 0
start = 0
end = 5000
target_depth = 0
depth_success = 0
depth_needed = 10
depth_last_i = 200
invalid_token_n = 0
for i in range(start, end):
    optimizer.zero_grad()
    # if i > 299 and (i % 100 == 0):
    #         temperature /= 1.03
    tokens, tokens_logs = sample_sequence(model, max_len, start_token_id, temperature, top_k, etw, vocab_size)
    if tokens.shape[0] <= 2 or ((start_token_id in tokens[1:]) or (torch.sum(tokens == end_token_id).item() != 1)):
        valid = 0.
    elif (4, 4) in zip(tokens.view(-1), tokens.view(-1)[1:]):
        valid = 0.
    elif i < 200:
        valid = 1. #if is_polish_normal_form([token_texts[t] for t in tokens]) else 0.
    else:
        valid = 1. if is_polish_normal_form([token_texts[t.item()] for t in tokens[1:-1]]) else 0.

    #print(f"{i} " + " ".join([token_texts[t.item()] for t in tokens]))
    tknst = [token_texts[t.item()] for t in tokens[1:-1]]
    # if valid > 0.5 and i > 200:
    #     print(f"{i} " + " ".join([token_texts[t.item()] for t in tokens]))

    if (i > 500 and len(tknst) == 1) or (i > 600 and len(tknst) == 2 and tknst[0] == 'not'):
        valid = 0.

    if (i > 1000 and valid < 0.5):
        invalid_num += 1

    target = torch.full(tokens_logs.shape, valid)
    w = 1.
    if valid > 0.5 and i > 200:
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
        #add demorgan again if needed
        # if (i > 1000 and varn2 == 0) or (i > 2000 and varn2 ==1):
        #     w = 0
        #w = (i - last_hit) * max(varn2 - ((i-200)/100)**0.4, 0)
        # if invalid_token_n > 0:
        #     w = invalid_token_n
        #     invalid_token_n = 0
        w *= max(1 + (len(tknst) - 1 if target_depth == 0 else (2 * target_depth)), 1) * 3 #could devide by expected length

        if depth >= target_depth:
            depth_success += 1

        if depth_success == depth_needed:
            depth_needed = 10
            target_depth += 1
            depth_decay = (i - depth_last_i)
            depth_last_i = i
            depth_success = 0

        if depth < target_depth:
            w *= 2 ** (-2 * ((target_depth - depth -1) + min((i - depth_last_i)/depth_decay, 1)))
        else:
            w *= 2 ** (depth-target_depth)

        #w *= 1.5**((varn2 - 1 - (i)/800))

        assert w != 0
        last_hit = i
        print(f"{i} " + " ".join([token_texts[t.item()] for t in tokens[1:-1]]))
        print(f"{varn2} {depth} {exp1}")
    # elif i > 200:
    #     invalid_token_n += len(tknst)

    weight = torch.full(tokens_logs.shape, w / max(1, len(tknst)))

    loss = binary_cross_entropy_with_logits(weight=weight, input=tokens_logs, target=target)

    loss.backward()
    optimizer.step()

print (invalid_num/(end - start - 1000))
#sample
# model.eval()
# tokens_logs = sample_sequence(model, max_length, start_token_id, temperature, top_k, vocab_size)
# print([token_texts[t] for t in tokens_logs])


torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'state-5000.pt')