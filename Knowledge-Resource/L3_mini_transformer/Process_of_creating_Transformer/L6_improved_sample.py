# sample.py
# Safe sampling with temperature and top-k clamped to vocab size.

import argparse
import torch
from L3_model import MiniGPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="mini_gpt_best.pt")  # load best by default
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--max_new_tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=0)  # 0 = disabled
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    conf = ckpt["config"]
    itos = ckpt["itos"]
    vocab_size = conf["vocab_size"]

    model = MiniGPT(
        vocab_size=vocab_size,
        n_embed=conf["n_embed"],
        n_head=conf["n_head"],
        n_layer=conf["n_layer"],
        block_size=conf["block_size"],
        dropout=conf["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    @torch.no_grad()
    def decode(ids):
        return "".join(itos[int(i)] for i in ids)

    # Start from a random token (or replace with your own prompt ids)
    context = torch.randint(vocab_size, (1, 1), device=device)

    # --- safe top-k inside generate-like loop ---
    @torch.no_grad()
    def safe_generate(idx, max_new_tokens, temperature=1.0, top_k=0):
        top_k = int(top_k) if top_k is not None else 0
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.block_size :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k > 0:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    out = safe_generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
