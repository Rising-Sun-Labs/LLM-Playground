# train_minigpt_fixed.py
# Run in a fresh kernel/session to avoid stale compiled graphs.

import os, math, time, glob, urllib.request
from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

# ---------------------------
# Config (edit these toggles)
# ---------------------------
USE_BPE = False                 # set True to use BPE stream if tokenizer is present
TOKENIZER_PATH = 'bpe/tokenizer.json'
DATA_DIR = None                 # e.g., '/path/to/text_or_code_corpus'
AUTO_DOWNLOAD_TINY_SHAKESPEARE = True

USE_AMP = torch.cuda.is_available()
USE_CKPT = True                 # gradient checkpointing: big memory saver
FORCE_COMPILE = False           # force attempt torch.compile (even with ckpt). If problems, set False.

BLOCK_SIZE = 256
VOCAB_SIZE_SYN = 8000           # used for synthetic stream only
BATCH_SIZE = 128 if torch.cuda.is_available() else 32
GRAD_ACCUM_STEPS = 2 if torch.cuda.is_available() else 1

LR = 3e-4
MAX_STEPS = 400
WARMUP_STEPS = 50
USE_COSINE = True
CLIP_NORM = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------
# Model: MiniGPT (decoder-only)
# --------------------------------
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
        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=-1)
        nh = self.n_head
        q = q.view(B,T,nh,-1).transpose(1,2)
        k = k.view(B,T,nh,-1).transpose(1,2)
        v = v.view(B,T,nh,-1).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
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
            nn.Linear(n_embed, 4*n_embed),
            nn.GELU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embed=384, n_head=6, n_layer=6, block_size=256,
                 dropout=0.1, grad_checkpointing=False):
        super().__init__()
        self.block_size = block_size
        self.grad_checkpointing = grad_checkpointing
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
        x = self.tok_emb(idx) + self.pos_emb(pos)[None,:,:]
        x = self.drop(x)
        if self.grad_checkpointing and self.training:
            import torch.utils.checkpoint as ckpt
            if not x.requires_grad:
                x.requires_grad_(True)
            # Wrap module in a lambda to avoid Dynamo+ckpt weirdness
            for blk in self.blocks:
                x = ckpt.checkpoint(lambda y: blk(y), x, use_reentrant=False)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

# --------------------------------
# Data: synthetic or BPE stream
# --------------------------------
def make_stream(n_tokens:int, vocab_size:int):
    return torch.randint(0, vocab_size, (n_tokens,), dtype=torch.long)

def get_batch(stream: torch.Tensor, block_size:int, batch_size:int, device='cpu'):
    hi = len(stream) - block_size - 1
    if hi <= 0:
        raise RuntimeError(f"Stream too small for block_size {block_size}. len={len(stream)}")
    idx = torch.randint(0, hi, (batch_size,))
    x = torch.stack([stream[i:i+block_size] for i in idx])
    y = torch.stack([stream[i+1:i+1+block_size] for i in idx])
    return x.to(device), y.to(device)

def maybe_build_bpe_stream() -> Optional[tuple]:
    if not USE_BPE or not os.path.exists(TOKENIZER_PATH):
        return None
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    eos_id = tok.token_to_id('<eos>')
    texts: List[str] = []

    if DATA_DIR and os.path.isdir(DATA_DIR):
        for ext in ['*.txt','*.md','*.py','*.js','*.ts','*.java','*.go','*.rs','*.c','*.cpp']:
            for p in glob.glob(os.path.join(DATA_DIR, '**', ext), recursive=True):
                try:
                    s = open(p, 'r', encoding='utf-8', errors='ignore').read()
                    if len(s) >= 100:
                        texts.append(s)
                except Exception:
                    pass

    if not texts and AUTO_DOWNLOAD_TINY_SHAKESPEARE:
        os.makedirs('data', exist_ok=True)
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, 'data/input.txt')
        texts.append(open('data/input.txt','r',encoding='utf-8').read())
        print('[BPE] Downloaded tiny Shakespeare as fallback')

    if not texts:
        print('[BPE] No texts found; falling back to synthetic stream.')
        return None

    ids = []
    for t in texts:
        enc = tok.encode(t).ids
        if enc:
            ids.extend(enc)
            if eos_id is not None:
                ids.append(eos_id)
    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_stream, val_stream = data[:n], data[n:]
    vocab_size = tok.get_vocab_size()
    print(f'[BPE] Using BPE stream | vocab_size={vocab_size} | lengths: {len(train_stream)}, {len(val_stream)}')
    return train_stream, val_stream, vocab_size

bpe = maybe_build_bpe_stream()
if bpe is None:
    VOCAB_SIZE = VOCAB_SIZE_SYN
    train_stream = make_stream(200_000, VOCAB_SIZE)
    val_stream   = make_stream(20_000,  VOCAB_SIZE)
else:
    train_stream, val_stream, VOCAB_SIZE = bpe

# --------------------------------
# Build model (guard torch.compile)
# --------------------------------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('medium')  # 'high' is also fine
    except Exception:
        pass

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    n_embed=384, n_head=6, n_layer=6,
    block_size=BLOCK_SIZE, dropout=0.1,
    grad_checkpointing=USE_CKPT
).to(device)

# By default, avoid compile when checkpointing (very reliable).
# If you want to try both, set FORCE_COMPILE=True.
if FORCE_COMPILE:
    try:
        model = torch.compile(model, mode='max-autotune')
        print('torch.compile: enabled (forced)')
    except Exception as e:
        print('torch.compile: disabled →', repr(e))
else:
    if not USE_CKPT and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='max-autotune')
            print('torch.compile: enabled (no ckpt)')
        except Exception as e:
            print('torch.compile: disabled →', repr(e))
    else:
        print('torch.compile: skipped (ckpt on)')

# --------------------------------
# Train utils
# --------------------------------
def lr_factor(step: int) -> float:
    if step < WARMUP_STEPS:
        return max(1e-8, (step+1)/max(1, WARMUP_STEPS))
    if not USE_COSINE:
        return 1.0
    progress = (step - WARMUP_STEPS)/max(1, MAX_STEPS - WARMUP_STEPS)
    min_factor = 0.1
    return min_factor + 0.5*(1-min_factor)*(1 + math.cos(math.pi*progress))

def eval_loss(n_iter=10) -> float:
    model.eval()
    s = 0.0
    with torch.no_grad():
        for _ in range(n_iter):
            xb, yb = get_batch(val_stream, BLOCK_SIZE, BATCH_SIZE, device)
            _, loss = model(xb, yb)
            s += float(loss.item())
    model.train()
    return s / max(1, n_iter)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler(enabled=USE_AMP)

# --------------------------------
# Warmup few steps (stabilize kernels)
# --------------------------------
for _ in range(3):
    xb, yb = get_batch(train_stream, BLOCK_SIZE, BATCH_SIZE, device)
    dtype = torch.bfloat16 if (USE_AMP and torch.cuda.is_available()) else torch.float32
    with autocast(enabled=USE_AMP, dtype=dtype if USE_AMP else None):
        _, loss = model(xb, yb)
    loss.backward()
    opt.zero_grad(set_to_none=True)

# --------------------------------
# Training loop
# --------------------------------
if torch.cuda.is_available(): torch.cuda.synchronize()
t0 = time.perf_counter()
tokens_processed = 0

for step in range(MAX_STEPS):
    # set per-step LR
    fac = lr_factor(step)
    for pg in opt.param_groups: pg['lr'] = LR * fac

    opt.zero_grad(set_to_none=True)
    micro_bs = max(1, BATCH_SIZE // max(1, GRAD_ACCUM_STEPS))
    for _ in range(GRAD_ACCUM_STEPS):
        xb, yb = get_batch(train_stream, BLOCK_SIZE, micro_bs, device)
        dtype = torch.bfloat16 if (USE_AMP and torch.cuda.is_available()) else torch.float32
        with autocast(enabled=USE_AMP, dtype=dtype if USE_AMP else None):
            _, loss = model(xb, yb)
            loss = loss / max(1, GRAD_ACCUM_STEPS)
        scaler.scale(loss).backward()
        tokens_processed += xb.numel()

    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    scaler.step(opt); scaler.update()

    if (step+1) % 100 == 0 or (step+1) == MAX_STEPS:
        vl = eval_loss(5)
        print(f"step {step+1}/{MAX_STEPS} | val_loss {vl:.4f}")

if torch.cuda.is_available(): torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
tok_per_sec = int(tokens_processed / max(elapsed, 1e-6))
print(f"Elapsed: {elapsed:.2f}s | Tokens processed: {tokens_processed:,} | ~{tok_per_sec:,} tok/s")
