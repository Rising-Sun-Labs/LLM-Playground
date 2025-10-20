# train.py
# Mini GPT training with warmup+cosine LR, AMP, grad accumulation, and best-ckpt saving.

import math
import argparse
import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import trange

from L2_data import CharDataset
from L3_model import MiniGPT


def main():
    p = argparse.ArgumentParser()
    # Data / device
    p.add_argument("--data", type=str, default="input.txt")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    # Model
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_embed", type=int, default=256)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training knobs
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--val_every", type=int, default=200)
    # Schedules / stability
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--use_cosine", action="store_true")
    # Saving
    p.add_argument("--save", type=str, default="mini_gpt_last.pt")
    p.add_argument("--save_best", type=str, default="mini_gpt_best.pt")
    args = p.parse_args()

    # Device fallback
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    # Dataset & model
    ds = CharDataset(args.data, block_size=args.block_size, device=device)
    model = MiniGPT(
        vocab_size=ds.vocab_size,
        n_embed=args.n_embed,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        dropout=args.dropout,
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(args.device == "cuda"))

    # LR schedule helpers
    def lr_factor(step: int) -> float:
        # Linear warmup
        if step < args.warmup_steps:
            return max(1e-8, (step + 1) / max(1, args.warmup_steps))
        if not args.use_cosine:
            return 1.0
        # Cosine decay to 10% of base LR
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        min_factor = 0.1
        return min_factor + 0.5 * (1 - min_factor) * (1 + math.cos(math.pi * progress))

    def set_lr(step: int):
        factor = lr_factor(step)
        for pg in opt.param_groups:
            pg["lr"] = args.lr * factor

    # Eval helper
    @torch.no_grad()
    def eval_loss(iters=50) -> float:
        model.eval()
        total = 0.0
        for _ in range(iters):
            x, y = ds.get_batch("val", args.batch_size)
            _, loss = model(x, y)
            total += float(loss.item())
        model.train()
        return total / max(1, iters)

    # Train
    model.train()
    best_vl = float("inf")

    for step in trange(args.max_steps, desc="training"):
        set_lr(step)

        opt.zero_grad(set_to_none=True)

        # Gradient accumulation
        micro_bs = max(1, args.batch_size // max(1, args.grad_accum_steps))
        for _ in range(args.grad_accum_steps):
            x, y = ds.get_batch("train", micro_bs)
            with autocast(enabled=(args.device == "cuda")):
                _, loss = model(x, y)
                loss = loss / max(1, args.grad_accum_steps)
            scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        # Periodic validation
        if (step + 1) % args.val_every == 0 or (step + 1) == args.max_steps:
            vl = eval_loss(iters=20)
            ppl = math.exp(vl) if vl < 20 else float("inf")
            print(f"\nstep {step+1} | val_loss {vl:.4f} | ppl ~ {ppl:.2f}")

            # Save best
            if vl < best_vl:
                best_vl = vl
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": {
                            "vocab_size": ds.vocab_size,
                            "n_embed": args.n_embed,
                            "n_head": args.n_head,
                            "n_layer": args.n_layer,
                            "block_size": args.block_size,
                            "dropout": args.dropout,
                        },
                        "itos": ds.itos,
                    },
                    args.save_best,
                )
                print(f"new best! saved {args.save_best}")

    # Save last
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "vocab_size": ds.vocab_size,
                "n_embed": args.n_embed,
                "n_head": args.n_head,
                "n_layer": args.n_layer,
                "block_size": args.block_size,
                "dropout": args.dropout,
            },
            "itos": ds.itos,
        },
        args.save,
    )
    print(f"saved last checkpoint to {args.save}")


if __name__ == "__main__":
    main()
