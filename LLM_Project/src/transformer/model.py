
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B,T,C = x.size()
        qkv = self.qkv(x)
        q,k,v = qkv.split(C, dim=2)
        q = q.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        k = k.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        v = v.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            att = att.masked_fill(attn_mask==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=4, d_mlp=1024, max_len=256, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model,n_heads,d_mlp,dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx):
        B,T = idx.shape
        pos = torch.arange(0,T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = torch.tril(torch.ones(T,T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v,_ = torch.topk(logits, top_k)
                logits[logits < v[:,[-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
