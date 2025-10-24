import argparse, sys
from harness.runner import run_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", default="miniLM.pt")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    path, metrics, cfg = run_all(args.config, args.ckpt, args.tokenizer, device=args.device)
    print("Wrote:", path)
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    sys.exit(main())
