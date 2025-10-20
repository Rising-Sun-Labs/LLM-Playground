import os, math, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from lib_minigpt import MiniGPT, make_stream, TokenStreamDataset

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
    return min_factor + 0.5*(1-min_factor)*(1 + math.cos(math.pi*progress))

def main():
    setup_dist()
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK','0')}" if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try: torch.set_float32_matmul_precision('medium')
        except Exception: pass

    BLOCK = int(os.environ.get('BLOCK_SIZE', '256'))
    VOCAB = int(os.environ.get('VOCAB_SIZE', '8000'))
    TRAIN_TOK = int(os.environ.get('TRAIN_TOKENS', '600000'))
    VAL_TOK   = int(os.environ.get('VAL_TOKENS',   '60000'))
    BATCH = int(os.environ.get('BATCH_SIZE', '32'))
    ACCUM = int(os.environ.get('GRAD_ACCUM', '2'))
    LR    = float(os.environ.get('LR', '3e-4'))
    MAX_STEPS = int(os.environ.get('MAX_STEPS', '300'))
    WARMUP = int(os.environ.get('WARMUP_STEPS', '100'))
    CLIP = float(os.environ.get('CLIP_NORM', '1.0'))
    USE_AMP = torch.cuda.is_available()

    train_stream = make_stream(TRAIN_TOK, VOCAB)
    val_stream   = make_stream(VAL_TOK,   VOCAB)
    train_ds = TokenStreamDataset(train_stream, BLOCK)
    val_ds   = TokenStreamDataset(val_stream,   BLOCK)
    train_samp = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_samp   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=train_samp, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, sampler=val_samp,   num_workers=2, pin_memory=True)

    model = MiniGPT(VOCAB, n_embed=384, n_head=6, n_layer=6, block_size=BLOCK, dropout=0.1).to(device)
    model = DDP(model, device_ids=[device] if device.type == 'cuda' else None)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler(enabled=USE_AMP)

    def run_val():
        model.eval()
        total, count = torch.tensor(0.0, device=device), torch.tensor(0, device=device)
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb)
                total += loss.detach(); count += 1
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        model.train()
        return (total / count).item() if int(count.item())>0 else float('nan')

    # Tiny warmup
    for _ in range(2):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            with autocast(enabled=USE_AMP, dtype=torch.bfloat16 if USE_AMP else None):
                _, loss = model(xb, yb)
            loss.backward(); opt.zero_grad(set_to_none=True); break

    step = 0
    while step < MAX_STEPS:
        train_samp.set_epoch(step)
        for xb, yb in train_dl:
            fac = cosine_factor(step, MAX_STEPS, warmup=WARMUP, min_factor=0.1)
            for pg in opt.param_groups: pg['lr'] = LR * fac
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP, dtype=torch.bfloat16 if USE_AMP else None):
                _, loss = model(xb.to(device, non_blocking=True), yb.to(device, non_blocking=True))
                loss = loss / max(1, ACCUM)
            scaler.scale(loss).backward()
            if (step+1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                scaler.step(opt); scaler.update()
            step += 1
            if step % 100 == 0 or step == MAX_STEPS:
                if is_main():
                    vl = run_val()
                    print(f"step {step}/{MAX_STEPS} | val_loss {vl:.4f}")
            if step >= MAX_STEPS: break

    if is_main():
        torch.save(model.module.state_dict(), 'minigpt_ddp.pt')
    dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__':
    main()
