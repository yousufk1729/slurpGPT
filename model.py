import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, embed_size, head_size, block_size, dropout):
        super().__init__()
        # K,Q,V are projected from embedding dimension (B,T,embed_dim) to head_size (B,T,head_dim)
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        # Register casual mask (for autoregressive behaviour) so optimizer doesn't change it 
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        # K,Q,V are projected from embedding dimension (B,T,embed_dim) to head_size (B,T,head_dim)
        k = self.key(x)    # (B, T, head_dim)
        q = self.query(x)  # (B, T, head_dim)
        v = self.value(x)  # (B, T, head_dim)
        # Compute attention scores
        # scores = Q * K^T * 1/sqrt(d_k)
        # (B, T, head_dim) @ (B, head_dim, T) -> (B, T, T)
        scores = (q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5)
        # Mask
        scores = scores.masked_fill(self.tril[:seq_len, :seq_len] == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)  # (B, T, T)
        attn_weights = self.dropout(attn_weights)
        # Apply attention to values
        # (B, T, T) @ (B, T, head_dim) -> (B, T, head_dim)
        attn_output = attn_weights @ v
        return attn_output

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_size, head_size, block_size, dropout) for _ in range(num_heads)])
        # Projects concatenated head outputs back to original embedding dimension
        # (B, T, head_dim * num_heads) -> (B, T, embed_dim)
        # Dimensions are, in fact, the same, but there is still a projection 
        # Because the nature of the 2 spaces is different
        self.proj = nn.Linear(head_size * num_heads, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input: (B, T, embed_dim)
        # concat (B, T, head_dim) + (B, T, head_dim) + ... (B, T, head_dim) -> (B,T, num_heads * head_size) 
        multi_head_output = torch.cat([head(x) for head in self.heads], dim=-1)
        # Project back to embedding dimension: (B, T, embed_dim)
        output = self.dropout(self.proj(multi_head_output))
        return output

class FeedForward(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # Specifications according to paper
            nn.Linear(embed_size, 4 * embed_size), 
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embed_size, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(embed_size, num_heads, head_size, block_size, dropout)
        self.ffwd = FeedForward(embed_size, dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # Residual connections, paper is post-norm, Karpathy uses pre-norm
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, head_size, num_layers, block_size, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[Block(embed_size, num_heads, head_size, block_size, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_size)  
        # Produces logits for every token
        self.demb = nn.Linear(embed_size, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        # Embeddings: both (B, T, embed_dim)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(seq_len, device=self.device))
        x = tok_emb + pos_emb  
        x = self.blocks(x)      
        x = self.ln_f(x)       
        # Project to vocabulary space for logits
        # (B, T, embed_dim) -> B, T, vocab_size)
        logits = self.demb(x) 
        if targets is None:
            loss = None
        else:
            # Flatten logits and targets: (B,T,vocab_size) to (B*T, vocab_size) and (B,T) to (B*T,)
            batch_size_flat = batch_size * seq_len
            logits_flat = logits.view(batch_size_flat, -1)
            targets_flat = targets.view(batch_size_flat)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # Get predictions (B, T, vocab_size)
            logits, _ = self(idx_cond)  
            # Focus only on the last time step (B, vocab_size)
            logits = logits[:, -1, :] 
            # Softmax, then sample (B, vocab_size) -> (B, 1)
            probs = F.softmax(logits, dim=-1)  # 
            idx_next = torch.multinomial(probs, num_samples=1)  
            # Append sampled index to the running sequence (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx