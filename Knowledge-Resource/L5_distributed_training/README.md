# Distributed / Multi-GPU Training (PyTorch DDP) — Step by Step

This guide shows how to scale your training from 1 GPU → many GPUs (and many machines) with **PyTorch DistributedDataParallel (DDP)**. You’ll get:

- A mental model (what actually changes vs single-GPU)
- Minimal working examples
- A **MiniGPT‐style** training loop adapted to DDP
- Launch commands for 1 node / N nodes
- Mixed precision, gradient accumulation, grad checkpointing, and saving checkpoints correctly
- Troubleshooting tips you’ll actually use

---

## 0) TL;DR (What changes for DDP?)

Only a few things:

1. **Launch** with `torchrun` (or `python -m torch.distributed.run`) so each GPU gets its own process.  
2. **Initialize process group** in your script.  
3. **Pin device** (each process uses one GPU).  
4. **Wrap model with `DistributedDataParallel` (DDP)**.  
5. **Use `DistributedSampler`** for your dataset.  
6. **Log & save only on rank 0** to avoid spam and file corruption.

That’s it. Your model, loss, optimizer, AMP, grad-accum, checkpointing… stay essentially the same.

---

## 1) Install & Environment

- **NVIDIA (Linux/Windows):** Use the **NCCL** backend (default for CUDA)  
- **CPU / macOS:** Use **GLOO** backend

```bash
pip install torch torchvision torchaudio
```

### Sanity (see your GPUs)
```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
PY
```

---

## 2) Launching DDP

### Single machine with N GPUs
```bash
# Use ALL GPUs
torchrun --standalone --nnodes=1 --nproc_per_node=$(nvidia-smi -L | wc -l) train_ddp.py

# Or pick GPUs 0,1 only
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train_ddp.py
```

### Multi-node (e.g., 2 machines, each with 4 GPUs)
Choose a host (node 0) and a free port, find its IP.

On **node 0**:
```bash
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=29500
NNODES=2
NPERNODE=4

torchrun --nnodes=$NNODES --nproc_per_node=$NPERNODE   --rdzv_backend=c10d   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT   train_ddp.py
```

On **node 1** (other machines), run the **same command** using the same `MASTER_ADDR:MASTER_PORT`.

> Tip: firewalls must allow `$MASTER_PORT`. For Slurm, consider `torchrun --standalone` replacements or SLURM envs.

---

## 3) Minimal DDP Skeleton

```python
# train_ddp.py
import os, math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def setup_dist():
    # torchrun sets these:
    # RANK, WORLD_SIZE, LOCAL_RANK
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def cleanup_dist():
    dist.barrier()
    dist.destroy_process_group()

def is_main():
    # Global rank 0 only
    return int(os.environ.get("RANK", "0")) == 0

def main():
    setup_dist()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if torch.cuda.is_available() else "cpu")

    # Dummy data
    x = torch.randn(10000, 128)
    y = torch.randint(0, 10, (10000,))
    ds = TensorDataset(x, y)

    # Important: DistributedSampler
    sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
    dl = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

    # Tiny model
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 512), torch.nn.ReLU(), torch.nn.Linear(512, 10)
    ).to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[device] if device.type == "cuda" else None)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)  # different shuffles across epochs
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        # Only rank 0 prints
        if is_main():
            print(f"Epoch {epoch+1} done")

    # Save only on rank 0
    if is_main():
        torch.save(model.module.state_dict(), "ddp_model.pt")

    cleanup_dist()

if __name__ == "__main__":
    main()
```

**Key bits:**
- `DistributedSampler` ensures each process sees a unique shard.
- `sampler.set_epoch(epoch)` gives different shuffles each epoch.
- `DDP(model, device_ids=[local_device])` for CUDA per process.
- Save checkpoints from **rank 0**: `model.module.state_dict()`.

---

## 4) Adapting Your MiniGPT Training to DDP

Below is the **diff you need** to move your MiniGPT loop to DDP. It assumes you already have: model building, AMP, gradient accumulation, cosine LR, etc.

### 4.1 Init & device

```python
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_dist():
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def is_main():
    return int(os.environ.get("RANK", "0")) == 0

setup_dist()
device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if torch.cuda.is_available() else "cpu")
```

### 4.2 Dataset/DataLoader

If you use a **streaming** batch maker (like your `get_batch` that draws from a big tensor), you can **keep it**, but ensure each rank samples different regions to avoid accidental overlap. The most robust path is to use a **`DistributedSampler`** when you have a standard `Dataset`. For a streaming tensor, a simple trick:

- Pass a `global_seed = base_seed + rank` into your random index generation, or
- Offset your random start by `rank * offset_stride` to decorrelate.

If you switch to a `Dataset`:

```python
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class TokenStreamDataset(Dataset):
    def __init__(self, tensor_data, block_size):
        self.data = tensor_data
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size - 1
    def __getitem__(self, i):
        x = self.data[i:i+self.block_size]
        y = self.data[i+1:i+1+self.block_size]
        return x, y

train_ds = TokenStreamDataset(train_stream, BLOCK_SIZE)
val_ds   = TokenStreamDataset(val_stream, BLOCK_SIZE)

train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)

train_loader = DataLoader(train_ds, batch_size=micro_bs, sampler=train_sampler, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=micro_bs, sampler=val_sampler,   num_workers=4, pin_memory=True)
```

> If you keep your `get_batch` stream function, add a **per-rank RNG** (`torch.Generator(device=device).manual_seed(base+rank)`) to draw indices differently on each rank. This is simpler and often fine for LMs.

### 4.3 Wrap the model, keep AMP, grad-accum & checkpointing

```python
model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    n_embed=384, n_head=6, n_layer=6,
    block_size=BLOCK_SIZE, dropout=0.1,
    grad_checkpointing=True   # keep if you already use it
).to(device)

# Wrap with DDP (for CUDA: device_ids=[device])
model = DDP(model, device_ids=[device] if device.type == "cuda" else None)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
```

### 4.4 Training loop (unchanged except: sampler epoch, rank 0 logging)

```python
for epoch in range(NUM_EPOCHS):
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    for step, (xb, yb) in enumerate(train_loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.bfloat16 if USE_AMP else None):
            _, loss = model(xb, yb)
            loss = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)

    if is_main():
        print(f"epoch {epoch+1} done")
```

### 4.5 Validation and reducing metrics

Example: average `val_loss` across ranks:

```python
@torch.no_grad()
def eval_loss_ddp(val_loader):
    model.eval()
    total, count = torch.tensor(0.0, device=device), torch.tensor(0, device=device)
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        total += loss.detach()
        count += 1
    # All-reduce (sum) across ranks
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    model.train()
    return (total / count).item()
```

### 4.6 Saving & loading checkpoints (rank 0 only)

```python
if is_main():
    torch.save({
        "model": model.module.state_dict(),  # unwrap DDP
        "opt": opt.state_dict(),
        "step": global_step,
        "config": {...},
    }, "ckpt.pt")
```

Loading:
```python
ckpt = torch.load("ckpt.pt", map_location="cpu")
model.module.load_state_dict(ckpt["model"])  # after wrapping with DDP
opt.load_state_dict(ckpt["opt"])
```

---

## 5) Mixed Precision, Grad Accum, Grad Checkpointing

- **AMP:** Works as usual. Keep `GradScaler`, wrap forward pass in `autocast`.
- **Grad Accumulation:** Same logic; each rank accumulates gradients locally, then steps.  
- **Grad Checkpointing:** Works with DDP. If you use `torch.utils.checkpoint.checkpoint_sequential`, it’s very stable.

> **Don’t** mix DDP with `torch.compile` on macOS or any path with spaces—Inductor C++ often breaks. You can scale great without compile.

---

## 6) Multi-Node networking notes

- Use **consistent** `MASTER_ADDR`, `MASTER_PORT`, `--nnodes`, `--nproc_per_node` on all nodes.
- NCCL needs open ports and no firewall blocking.  
- If you get NCCL hangs:
  - `export NCCL_DEBUG=INFO`
  - `export NCCL_IB_DISABLE=1` (if no Infiniband)
  - `export NCCL_P2P_DISABLE=1` (some clouds)
  - `export NCCL_SOCKET_IFNAME=eth0` (or your interface)

---

## 7) Example: End-to-End MiniGPT DDP Trainer

This is a compact, **ready-to-run** DDP version using a `Dataset` wrapper. Save as `train_ddp_minigpt.py` and launch with `torchrun` as shown earlier.

```python
# train_ddp_minigpt.py
import os, math, time, glob, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

USE_AMP = torch.cuda.is_available()
BLOCK_SIZE = 256
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2
LR = 3e-4
EPOCHS = 3
CLIP_NORM = 1.0

def setup_dist():
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def is_main():
    return int(os.environ.get("RANK", "0")) == 0

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, block_size, drop=0.1):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embed, 3*n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)
        self.attn_drop, self.resid_drop = nn.Dropout(drop), nn.Dropout(drop)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1,1,block_size,block_size))
    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        nh = self.n_head
        q = q.view(B,T,nh,-1).transpose(1,2)
        k = k.view(B,T,nh,-1).transpose(1,2)
        v = v.view(B,T,nh,-1).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / (k.size(-1) ** 0.5)
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = self.attn_drop(att.softmax(dim=-1))
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,-1)
        return self.resid_drop(self.proj(y))

class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, drop=0.1):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(n_embed), CausalSelfAttention(n_embed, n_head, block_size, drop)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(nn.Linear(n_embed, 4*n_embed), nn.GELU(), nn.Linear(4*n_embed, n_embed), nn.Dropout(drop))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab, n_embed=384, n_head=6, n_layer=6, block=256, drop=0.1):
        super().__init__()
        self.block_size = block
        self.tok_emb = nn.Embedding(vocab, n_embed)
        self.pos_emb = nn.Embedding(block, n_embed)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, block, drop) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab, bias=False)
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

class TokenStreamDataset(Dataset):
    def __init__(self, tensor, block):
        self.data, self.block = tensor, block
    def __len__(self): return len(self.data) - self.block - 1
    def __getitem__(self, i):
        x = self.data[i:i+self.block]
        y = self.data[i+1:i+1+self.block]
        return x, y

def main():
    setup_dist()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    # Fake data stream (replace with your BPE tensor)
    VOCAB = 8000
    train_stream = torch.randint(0, VOCAB, (500_000,), dtype=torch.long)
    val_stream   = torch.randint(0, VOCAB, (50_000,),  dtype=torch.long)

    train_ds = TokenStreamDataset(train_stream, BLOCK_SIZE)
    val_ds   = TokenStreamDataset(val_stream,   BLOCK_SIZE)

    train_samp = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_samp   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)

    # Micro-batch per step; use GRAD_ACCUM_STEPS to enlarge global batch
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_samp, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, sampler=val_samp,   num_workers=2, pin_memory=True)

    model = MiniGPT(VOCAB, n_embed=384, n_head=6, n_layer=6, block=BLOCK_SIZE, drop=0.1).to(device)
    model = DDP(model, device_ids=[device] if device.type == "cuda" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler(enabled=USE_AMP)

    def run_val():
        model.eval()
        total, count = torch.tensor(0.0, device=device), torch.tensor(0, device=device)
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb)
                total += loss.detach()
                count += 1
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        model.train()
        return (total / count).item()

    for epoch in range(EPOCHS):
        train_samp.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(train_dl):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with autocast(enabled=USE_AMP, dtype=torch.bfloat16 if USE_AMP else None):
                _, loss = model(xb, yb)
                loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

        if is_main():
            vl = run_val()
            print(f"epoch {epoch+1}/{EPOCHS} | val_loss {vl:.4f}")

    if is_main():
        torch.save(model.module.state_dict(), "minigpt_ddp.pt")

    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Launch on 1 machine with all GPUs:
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=$(nvidia-smi -L | wc -l) train_ddp_minigpt.py
```

---

## 8) Troubleshooting (bookmark this)

- **Inductor/`torch.compile` errors (macOS / paths with spaces):** keep compile **off**. You don’t need it to scale.  
- **Hang at init:** firewalls; set `MASTER_ADDR/PORT` correctly; try `NCCL_DEBUG=INFO NCCL_IB_DISABLE=1`.  
- **Only 1 GPU gets used:** don’t call `DataParallel` inside DDP; ensure `torchrun --nproc_per_node=NUM_GPUS`.  
- **Duplicate logging / multiple checkpoints:** print/save **only on rank 0**.  
- **Shuffles look identical across ranks:** you forgot `DistributedSampler` or `sampler.set_epoch(epoch)`.  
- **OOM after scaling GPUs:** your **per-GPU** batch might still be too large; reduce `BATCH_SIZE` or increase `GRAD_ACCUM_STEPS`.  
- **AMP overflow:** lower LR, check gradients after `scaler.unscale_(opt)`, or temporarily disable AMP to debug.

---

## 9) What about FSDP / ZeRO?

- **DDP** replicates full model on each GPU (simple, strong baseline).  
- **FSDP** shards params/gradients/optimizer states across GPUs → scale much larger models. Start with DDP; move to FSDP when memory is your bottleneck.

## 9.1) When to use:

- Model doesn't fit on one GPU (even with small batch)
- You need to raise sequence length or hidden size and DDP OOMs.
---

## 10) Checklist before you scale

- [ ] Use `torchrun` to launch N processes (one per GPU)  
- [ ] `init_process_group()` in code  
- [ ] `DistributedSampler` (or per-rank RNG)  
- [ ] `DDP(model, device_ids=[local_device])`  
- [ ] AMP & grad-accum as usual  
- [ ] Save/log only on rank 0  
- [ ] Validate across ranks with all-reduce if you need a global metric
