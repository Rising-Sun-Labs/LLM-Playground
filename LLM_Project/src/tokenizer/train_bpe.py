import argparse
from pathlib import Path
from .simple_bpe import SimpleBPE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text = Path(args.text).read_text(encoding="utf-8")
    tok = SimpleBPE()
    tok.train(text, vocab_size=args.size, min_freq=2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    print(f"Saved tokenizer to {args.out} (vocab={len(tok.vocab)})")

if __name__ == "__main__":
    main()
