# sample.py
import argparse, torch
from L3_model import MiniGPT

p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=str, default="mini_gpt.pt")
p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda","mps"])
p.add_argument("--max_new_tokens", type=int, default=400)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=200)
args = p.parse_args()

device = torch.device(args.device if (args.device!="cuda" or torch.cuda.is_available()) else "cpu")
ckpt = torch.load(args.ckpt, map_location=device)

conf = ckpt["config"]
itos = ckpt["itos"]
vocab_size = conf["vocab_size"]

model = MiniGPT(
    vocab_size=vocab_size,
    n_embed=conf["n_embed"], n_head=conf["n_head"], n_layer=conf["n_layer"],
    block_size=conf["block_size"], dropout=conf["dropout"]
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

@torch.no_grad()
def decode(ids):
    return "".join(itos[int(i)] for i in ids)

context = torch.randint(vocab_size, (1,1), device=device)
out = model.generate(context, max_new_tokens=args.max_new_tokens,
                     temperature=args.temperature, top_k=args.top_k)
print(decode(out[0].tolist()))
