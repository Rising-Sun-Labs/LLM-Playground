# export_torchscript.py
import torch, argparse
from model_def import MiniTransformerLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="miniLM.pt")
    ap.add_argument("--out", default="miniLM_ts.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    obj = torch.load(arg.ckpt, map_location=args.device)
    cfg = obj.get("config", {"vocab_size": 1200, "d_model": 256, "n_layers":6, "n_heads":4, "d_mlp":1024, "max_len":256})
    model = MiniTransformerLM(**cfg)
    sd = obj["model"] if "model" in obj else obj
    model.load_state_dict(sd); model.eval()

    example = torch.randint(0, cfg["vocab_size"], (1,8), dtype=torch.long)
    scripted = torch.jit.trace(model, example)
    scripted.save(args.out)

    print("Saved torchscript to",args.out)

if __name__ == "__main__":
    main()


# Run
# python export_torchscript.py --ckpt miniLM.pt --out miniLM_ts.pt
