# sample_bpe.py
# Load best checkpoint and generate from a prompt with safe temperature/top-k.

import argparse, torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

# --- Minimal model (must match train_bpe.py) ---

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embed, 3*n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1,1,block_size,block_size))
    def forward(self, x):
        B,T,C = x.shape
        qkv = self.qkv(x); q,k,v = qkv.chunk(3, dim=-1)
        nh = self.n_head
        q = q.view(B,T,nh,-1).transpose(1,2)
        k = k.view(B,T,nh,-1).transpose(1,2)
        v = v.view(B,T,nh,-1).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / (k.size(-1) ** 0.5)
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,-1)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed), nn.GELU(), nn.Linear(4*n_embed, n_embed), nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)
    def forward(self, idx):
        B,T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None,:,:]
        x = self.drop(x)
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        return self.head(x)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-8)
            if top_k and top_k > 0:
                k = min(int(top_k), logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# --- CLI ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bpe_best.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt", type=str, default="ROMEO:")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=80)
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device)

    conf = ck["config"]
    model = MiniGPT(
        vocab_size=conf["vocab_size"],
        n_embed=conf["n_embed"], n_head=conf["n_head"], n_layer=conf["n_layer"],
        block_size=conf["block_size"], dropout=conf["dropout"]
    ).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    tok = Tokenizer.from_file(ck.get("tokenizer_path", "bpe/tokenizer.json"))

    # encode prompt, generate, decode
    ctx = torch.tensor([tok.encode(args.prompt).ids], dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))

if __name__ == "__main__":
    main()
