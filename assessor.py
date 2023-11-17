import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Parameters for the model
vocab_size = 10  # This should be the size of your vocabulary
d_model = 512  # The dimension of the embeddings
nhead = 8  # Number of heads in the multi-head attention mechanisms
num_layers = 6  # Number of decoder layers
max_len = 50  # Maximum sequence length

# # Instantiate the model with a specific number of layers
model = Transformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, max_len=max_len)

# # Dummy inputs for the purpose of this example
# tgt = torch.randint(0, vocab_size, (5, 32))  # (sequence_length, batch_size)

# # Forward pass
# output = model(tgt)
# print(output.shape)  # (sequence_length, batch_size, vocab_size)

# import torch
# import torch.nn.functional as F

def top_k_logits(logits, k):
    if k == 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(1)
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

def sample_sequence(model, length, start_token_id, temperature=1.0, top_k=None, vocab_size=10):
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

    model.eval()
    with torch.no_grad():
        generated = torch.full((1, 1), start_token_id, dtype=torch.long, device=next(model.parameters()).device)
        for _ in range(length - 1):
            outputs = model(generated)  #(batch_num, vocab_size)
            next_token_logits = outputs[:, :] / temperature
            if top_k is not None:
                # next_token_logits = top_k_logits(next_token_logits, top_k)
                next_token_logits = torch.topk(next_token_logits, top_k, 1)[0]  #(batch_num, top_k)
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)   #(batch_size, 1)
            generated = torch.cat((generated, next_token), dim=0)
        return generated.squeeze()

# Parameters for inference
start_token_id = 1  # Assuming 1 is the ID for the start token in your vocabulary
length = 50  # The length of sequence to generate, including the start token
temperature = 0.7  # Can be adjusted for more or less randomness
top_k = 5  # Top-k filtering, should be less than the vocabulary size
vocab_size = 10  # Vocabulary size of the model

# Sample a sequence
sampled_sequence = sample_sequence(model, length, start_token_id, temperature, top_k, vocab_size)
print(sampled_sequence)
