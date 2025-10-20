import argparse, torch
from ..tokenizer.simple_bpe import SimpleBPE
from ..transformer.model import TinyGPT
TEMPLATE = "Summarize the following email in 2-3 sentences:\nEMAIL:\n{body}\nSUMMARY:"

def load_lm(ckpt_path, vocab_size, max_len, device):
    model = TinyGPT(vocab_size=vocab_size, max_len=max_len)
    obj = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(obj["model"] if "model" in obj else obj)
    model.to(device).eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = SimpleBPE.load(args.tokenizer)
    obj = torch.load(args.ckpt, map_location=args.device)
    vocab_size = obj.get("vocab_size", len(tok.vocab))
    max_len = obj.get("max_len", 256)

    model = load_lm(args.ckpt, vocab_size, max_len, args.device)
    prompt = TEMPLATE.format(body=args.text.strip())
    ids = tok.encode(prompt, add_special_bos_eos=True, max_len=max_len)
    x = torch.tensor([ids], dtype=torch.long, device=args.device)
    y = model.generate(x, max_new_tokens=120, temperature=0.9, top_k=40)
    print(tok.decode(y[0].tolist()))

if __name__ == "__main__":
    main()
