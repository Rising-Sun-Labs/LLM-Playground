# lesson 4 — training loop

# idea: minimize cross - entropy between predicted next token and the true next token.

import argparse, torch
from torch import optim
from tqdm import trange
from L2_data import CharDataset
from L3_model import MiniGPT


p = argparse.ArgumentParser()
p.add_argument("--data", type=str, default="input.txt")
p.add_argument("--device", type=str, default ="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda", "mps"])
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--block_size", type=int, default=128)
p.add_argument("--n_embed", type=int, default=256)
p.add_argument("--n_head", type=int, default=4)
p.add_argument("--n_layer", type = int, default=4)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--max_steps", type=int, default=2000)
p.add_argument("--val_every", type=int, default=200)
p.add_argument("--save", type=str, default="mini_gpt.pt")
args = p.parse_args()

device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

ds = CharDataset(args.data, block_size=args.block_size, device=device)

model = MiniGPT(
    vocab_size=ds.vocab_size,
    n_embed=args.n_embed, n_head=args.n_head, n_layer=args.n_layer,
    block_size=args.block_size, dropout=args.dropout
).to(device)


opt = optim.Adam(model.parameters(), lr=args.lr)

def eval_loss(split="val", iters=50):
    model.eval()
    import torch
    total = 0.0
    with torch.no_grad():
        for _ in range(iters):
            x, y = ds.get_batch("train", args.batch_size)
            _, loss = model(x, y)
            total += loss.item()
    model.train()
    return total / iters

for step in trange(args.max_steps, desc="training"):
    x, y = ds.get_batch("train", args.batch_size)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if (step+1) % args.val_every == 0:
        vl = eval_loss("val", iters = 20)
        print(f"\nstep {step+1} | train_loss {loss.item():.3f} | val_loss {vl:.3f} | val_loss {vl:.3f}\n")

torch.save({
    "model": model.state_dict(),
    "config":{
        "vocab_size": ds.vocab_size,
        "n_embed": args.n_embed, "n_head": args.n_head, "n_layer": args.n_layer,
        "block_size": args.block_size, "dropout": args.dropout
    },
    "itos": ds.itos
}, args.save)

print(f"saved {args.save}")