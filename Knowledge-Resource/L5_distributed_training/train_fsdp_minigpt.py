import os, math, time, torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from lib_minigpt import MiniGPT, make_stream, TokenStreamDataset, Block

def setup_dist():
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))

def is_main():
    return int(os.environ.get('RANK', '0')) == 0

def cosine_factor(step, max_steps, warmup=200, min_factor=0.1):
    if step < warmup:
        return max(1e-8, (step+1)/max(1, warmup))
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_factor + 0.5 * (1-min_factor) * (1 + math.cos(math.pi * progress))

def main():
    setup_dist()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try: torch.set_float32_matmul_precision('medium')
        except Exception: pass

    # Config
    BLOCK = int(os.environ.get('BLOCK_SIZE', '256'))
    VOCAB = int(os.environ.get('VOCAB_SIZE', '8000'))
    TRAIN_TOK = int(os.environ.get('TRAIN_TOKENS', '600000'))
    VAL_TOK   = int(os.environ.get('VAL_TOKENS',   '60000'))
    BATCH = int(os.environ.get('BATCH_SIZE', '32'))
    ACCUM = int(os.environ.get('GRAD_ACCUM', '2'))
    LR    = float(os.environ.get('LR', '3e-4'))
    MAX_STEPS = int(os.environ.get('MAX_STEPS', '500'))
    WARMUP = int(os.environ.get('WARMUP_STEPS', '200'))
    CLIP = float(os.environ.get('CLIP_NORM', '1.0'))
    USE_AMP = torch.cuda.is_available()

    # Data
    train_stream = make_stream(TRAIN_TOK, VOCAB)
    val_stream   = make_stream(VAL_TOK,   VOCAB)
    train_ds = TokenStreamDataset(train_stream, BLOCK)
    val_ds   = TokenStreamDataset(val_stream,   BLOCK)
    train_samp = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_samp   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=train_samp, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, sampler=val_samp,   num_workers=2, pin_memory=True)

    # Base model on CPU first (FSDP will move shards to GPU)
    base_model = MiniGPT(VOCAB, n_embed=384, n_head=6, n_layer=6, block_size=BLOCK, dropout=0.1)

    # Optional activation checkpointing on each transformer Block for memory
    for i, blk in enumerate(base_model.blocks):
        base_model.blocks[i] = checkpoint_wrapper(blk)

    # Auto-wrap policy for transformer Blocks
    auto_wrap = transformer_auto_wrap_policy({Block})

    # Mixed precision config for FSDP (use bf16 on CUDA if available)
    mp_policy = None
    if torch.cuda.is_available():
        from torch.distributed.fsdp import MixedPrecision
        mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)

    # Wrap with FSDP
    model = FSDP(base_model, auto_wrap_policy=auto_wrap, mixed_precision=mp_policy)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler(enabled=USE_AMP)

    def run_val():
        model.eval()
        total, count = torch.tensor(0.0, device=device), torch.tensor(0, device=device)
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(enabled=USE_AMP, dtype=torch.bfloat16 if USE_AMP else None):
                    _, loss = model(xb, yb)
                total += loss.detach()
                count += 1
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        model.train()
        return (total / count).item() if int(count.item())>0 else float('nan')

    # Warmup micro-steps
    for _ in range(3):
        for xb, yb in train_dl:
            with autocast(enabled=USE_AMP, dtype=torch.bfloat16 if USE_AMP else None):
                _, loss = model(xb.to(device), yb.to(device))
            loss.backward(); opt.zero_grad(set_to_none=True); break

    # Train
    step = 0
    while step < MAX_STEPS:
        train_samp.set_epoch(step)
        for xb, yb in train_dl:
            fac = cosine_factor(step, MAX_STEPS, warmup=WARMUP, min_factor=0.1)
            for pg in opt.param_groups: pg['lr'] = LR * fac
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP, dtype=torch.bfloat16 if USE_AMP else None):
                _, loss = model(xb.to(device, non_blocking=True), yb.to(device, non_blocking=True))
                loss = loss / 2  # ACCUM fixed to 2 for clarity
            scaler.scale(loss).backward()
            if (step+1) % 2 == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            step += 1
            if step % 100 == 0 or step == MAX_STEPS:
                if is_main():
                    vl = run_val()
                    print(f"step {step}/{MAX_STEPS} | val_loss {vl:.4f}")
            if step >= MAX_STEPS: break

    # Save full state dict on rank 0
    if is_main():
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            sd = model.state_dict()
        torch.save(sd, 'minigpt_fsdp_full.pt')

    dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__':
    main()
