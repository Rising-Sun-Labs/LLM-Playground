import math, torch
from torch import nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('mask', mask.view(1,1,block_size,block_size))
    def forward(self, x):
        B,T,C = x.shape
        qkv = self.qkv(x); q,k,v = qkv.chunk(3, dim=-1)
        nh = self.n_head
        q = q.view(B,T,nh,-1).transpose(1,2)
        k = k.view(B,T,nh,-1).transpose(1,2)
        v = v.view(B,T,nh,-1).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = self.attn_drop(att.softmax(dim=-1))
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,-1)
        return self.resid_drop(self.proj(y))

class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(nn.Linear(n_embed, 4*n_embed), nn.GELU(),
                                 nn.Linear(4*n_embed, n_embed), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embed=384, n_head=6, n_layer=6, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B,T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None,:,:])
        for blk in self.blocks: x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

def make_stream(n_tokens:int, vocab_size:int):
    return torch.randint(0, vocab_size, (n_tokens,), dtype=torch.long)

class TokenStreamDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_data, block_size):
        self.data = tensor_data
        self.block = block_size
    def __len__(self): return max(0, len(self.data) - self.block - 1)
    def __getitem__(self, i):
        x = self.data[i:i+self.block]
        y = self.data[i+1:i+1+self.block]
        return x, y
