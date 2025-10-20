# model.py
import math, torch, torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def build_rope_cache(T, Dh, base=10000.0, device="cpu"):
    half = Dh // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(T, device=device).float()
    freqs = torch.einsum("t,d->td", t, inv_freq)  # [T, Dh/2]
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # [T, Dh/2, 2]

def apply_rope(q, k, freqs):
    # q,k: [B, H, T, Dh], freqs: [T, Dh/2, 2]
    B,H,T,Dh = q.shape
    q = q.view(B,H,T,Dh//2,2)
    k = k.view(B,H,T,Dh//2,2)

    cos = freqs[:T,:,0].unsqueeze(0).unsqueeze(0)  # [1,1,T,half]
    sin = freqs[:T,:,1].unsqueeze(0).unsqueeze(0)

    q0, q1 = q[...,0], q[...,1]
    k0, k1 = k[...,0], k[...,1]

    q_rotated_0 = q0 * cos - q1 * sin
    q_rotated_1 = q0 * sin + q1 * cos
    k_rotated_0 = k0 * cos - k1 * sin
    k_rotated_1 = k0 * sin + k1 * cos

    q = torch.stack([q_rotated_0, q_rotated_1], dim=-1).reshape(B,H,T,Dh)
    k = torch.stack([k_rotated_0, k_rotated_1], dim=-1).reshape(B,H,T,Dh)
    return q, k


class MHA(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        assert d % heads == 0
        self.h = heads
        self.dk = d // heads
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rope_freqs):
        B,T,D = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        def sh(t): return t.view(B,T,self.h,self.dk).transpose(1,2)
        q,k,v = sh(q), sh(k), sh(v)
        q,k = apply_rope(q,k,rope_freqs[:T])
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.dk)
        mask = torch.triu(torch.full((T,T), float("-inf"), device=x.device), 1)
        att = att + mask
        p = att.softmax(-1)
        out = (p @ v).transpose(1,2).contiguous().view(B,T,D)
        return self.drop(self.proj(out))

class SwiGLU(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        inner = int(mult*d)
        self.w1 = nn.Linear(d, inner, bias=False)
        self.w2 = nn.Linear(d, inner, bias=False)
        self.w3 = nn.Linear(inner, d, bias=False)
    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self, d, heads, mlp_mult, dropout):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = MHA(d, heads, dropout)
        self.n2 = RMSNorm(d)
        self.mlp  = SwiGLU(d, mlp_mult)
    def forward(self, x, rope_freqs):
        x = x + self.attn(self.n1(x), rope_freqs)
        x = x + self.mlp(self.n2(x))
        return x

class MiniTransformerLM(nn.Module):
    def __init__(self, vocab, d=384, L=6, heads=6, mlp_mult=4, dropout=0.0, max_seq=1024):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.h = heads
        self.Dh = d // heads
        self.max_seq = max_seq
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads, mlp_mult, dropout) for _ in range(L)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.register_buffer("rope", build_rope_cache(max_seq, self.Dh, device="cpu"), persistent=False)

    def forward(self, idx):
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x, self.rope)
        x = self.norm(x)
        return self.head(x)

# Checkpoint: you see how RoPE is precomputed and applied; causal mask prevents peeking ahead.
