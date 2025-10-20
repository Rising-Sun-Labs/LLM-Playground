# train_bpe.py
# End-to-end mini GPT training with BPE tokenizer, cosine LR + warmup, AMP, and grad accumulation.

import os, math, glob, argparse, json
from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import trange

# ---------------------------
# Data loading (corpus)
# ---------------------------

def load_texts(data_dir: Optional[str], data_file: Optional[str], min_len: int = 100) -> List[str]:
    texts: List[str] = []
    if data_file:
        s = open(data_file, "r", encoding="utf-8", errors="ignore").read()
        if len(s) >= min_len:
            texts.append(s)
    elif data_dir:
        exts = ['*.txt','*.md','*.py','*.js','*.ts','*.java','*.go','*.rs','*.c','*.cpp']
        for ext in exts:
            for p in glob.glob(os.path.join(data_dir, '**', ext), recursive=True):
                try:
                    s = open(p, "r", encoding="utf-8", errors="ignore").read()
                    if len(s) >= min_len:
                        texts.append(s)
                except Exception:
                    pass
    else:
        # fallback tiny shakespeare if nothing provided
        os.makedirs("data", exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        import urllib.request
        urllib.request.urlretrieve(url, "data/input.txt")
        texts.append(open("data/input.txt", "r", encoding="utf-8").read())

    if not texts:
        raise RuntimeError("No usable text found. Provide --data_dir or --data_file with non-empty content.")
    return texts

# ---------------------------
# Tokenizer (Hugging Face tokenizers)
# ---------------------------

def maybe_train_tokenizer(texts: List[str], tok_path: str, vocab_size: int = 8000):
    if os.path.exists(tok_path):
        print(f"[tokenizer] using existing {tok_path}")
        return
    print(f"[tokenizer] training new BPE tokenizer at {tok_path} (vocab_size={vocab_size})")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    os.makedirs(os.path.dirname(tok_path) or ".", exist_ok=True)
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])
    tok.train_from_iterator(texts, trainer)
    tok.save(tok_path)

# ---------------------------
# Dataset
# ---------------------------

class BPEDataset:
    def __init__(self, texts: List[str], tokenizer_path: str, block_size: int = 256, device: Optional[str] = None):
        from tokenizers import Tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = Tokenizer.from_file(tokenizer_path)

        ids = []
        eos_id = self.tok.token_to_id("<eos>")
        for t in texts:
            enc = self.tok.encode(t).ids
            if enc:
                ids.extend(enc)
                if eos_id is not None:
                    ids.append(eos_id)

        data = torch.tensor(ids, dtype=torch.long)
        total_len = len(data)
        if total_len < 4:
            raise RuntimeError("Encoded corpus extremely small. Add more files or reduce block_size a lot.")

        # train/val split
        n = int(0.9 * total_len)
        self.train_data = data[:n]
        self.val_data   = data[n:]
        self.vocab_size = self.tok.get_vocab_size()

        # clamp block_size if needed
        if block_size >= len(self.train_data) - 1:
            new_bs = max(2, len(self.train_data) - 2)
            print(f"[BPEDataset] block_size {block_size} > train len; clamping to {new_bs}.")
            block_size = new_bs

        self.block_size = block_size

        if len(self.val_data) <= self.block_size + 1:
            print(f"[BPEDataset] Warning: very small val split for block_size={self.block_size} (len={len(self.val_data)}).")

    def get_batch(self, split: str, batch_size: int):
        src = self.train_data if split == "train" else self.val_data
        hi = len(src) - self.block_size - 1
        if hi <= 0:
            raise RuntimeError(
                f"Not enough {split} tokens for block_size={self.block_size}. len(src)={len(src)}. "
                "Add more data or reduce block_size."
            )
        idx = torch.randint(0, hi, (batch_size,))
        x = torch.stack([src[i:i + self.block_size] for i in idx])
        y = torch.stack([src[i + 1:i + 1 + self.block_size] for i in idx])
        return x.to(self.device), y.to(self.device)

# ---------------------------
# Model
# ---------------------------

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
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        nh = self.n_head
        q = q.view(B, T, nh, -1).transpose(1, 2)
        k = k.view(B, T, nh, -1).transpose(1, 2)
        v = v.view(B, T, nh, -1).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
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
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k and top_k > 0:
                k = min(int(top_k), logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# ---------------------------
# Training
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None, help="Folder of text/code/chats (recursive).")
    ap.add_argument("--data_file", type=str, default=None, help="Single text file.")
    ap.add_argument("--tokenizer_path", type=str, default="bpe/tokenizer.json")
    ap.add_argument("--vocab_size", type=int, default=8000)

    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--n_embed", type=int, default=384)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--grad_accum_steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--use_cosine", action="store_true")

    ap.add_argument("--val_every", type=int, default=200)
    ap.add_argument("--save_best", type=str, default="bpe_best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load data
    texts = load_texts(args.data_dir, args.data_file)
    # Train tokenizer if missing
    maybe_train_tokenizer(texts, args.tokenizer_path, vocab_size=args.vocab_size)

    # Dataset
    ds = BPEDataset(texts, tokenizer_path=args.tokenizer_path, block_size=args.block_size, device=str(device))
    print(f"vocab_size={ds.vocab_size} | block_size={ds.block_size} | train_len={len(ds.train_data)} | val_len={len(ds.val_data)}")

    # Model
    model = MiniGPT(
        vocab_size=ds.vocab_size,
        n_embed=args.n_embed, n_head=args.n_head, n_layer=args.n_layer,
        block_size=ds.block_size, dropout=args.dropout
    ).to(device)

    # Optim & AMP
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max(1e-8, (step+1)/max(1, args.warmup_steps))
        if not args.use_cosine:
            return 1.0
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        min_factor = 0.1
        return min_factor + 0.5*(1-min_factor)*(1 + math.cos(math.pi * progress))

    @torch.no_grad()
    def eval_loss(iters=20) -> float:
        model.eval()
        s = 0.0
        for _ in range(iters):
            x, y = ds.get_batch("val", args.batch_size)
            _, loss = model(x, y)
            s += float(loss.item())
        model.train()
        return s / max(1, iters)

    best = float("inf")
    for step in trange(args.max_steps, desc="training"):
        # set LR
        fac = lr_factor(step)
        for pg in opt.param_groups: pg["lr"] = args.lr * fac

        opt.zero_grad(set_to_none=True)
        micro_bs = max(1, args.batch_size // max(1, args.grad_accum_steps))

        # gradient accumulation
        for _ in range(args.grad_accum_steps):
            x, y = ds.get_batch("train", micro_bs)
            with autocast(enabled=(device.type == "cuda")):
                _, loss = model(x, y)
                loss = loss / max(1, args.grad_accum_steps)
            scaler.scale(loss).backward()

        # AMP-safe step + clip
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        if (step + 1) % args.val_every == 0 or (step + 1) == args.max_steps:
            vl = eval_loss(20)
            ppl = math.exp(vl) if vl < 20 else float("inf")
            print(f"\nstep {step+1} | val_loss {vl:.4f} | ppl ~ {ppl:.2f}")

            if vl < best:
                best = vl
                torch.save({
                    "model": model.state_dict(),
                    "config": {
                        "vocab_size": ds.vocab_size,
                        "n_embed": args.n_embed, "n_head": args.n_head, "n_layer": args.n_layer,
                        "block_size": ds.block_size, "dropout": args.dropout,
                    },
                    "tokenizer_path": args.tokenizer_path
                }, args.save_best)
                print(f"new best saved: {args.save_best}")

if __name__ == "__main__":
    main()
