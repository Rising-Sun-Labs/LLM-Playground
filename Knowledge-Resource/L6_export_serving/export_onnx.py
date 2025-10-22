# export_onnx.py
import torch, argparse
from model_def import MiniTransformerLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="miniLM.pt")
    ap.add_argument("--out", default="miniLM.onnx")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    obj = torch.load(args.ckpt, map_location=args.device)
    cfg = obj.get("config", {"vocab_size": 1200, "d_model":256, "n_layers":6, "n_heads": 4, "d_mlp":1024, "max_len":256})
    model = MiniTransformerLM(**cfg)
    sd = obj["model"] if "model" in obj else obj
    model.load_state_dict(sd); model.eval()


    dummy = torch.randint(0, cfg["vocab_size"], (1,8), dtype=torch.long)
    torch.onnx.export(
        model, (dummy,), args.out, input_names = ["input_ids"], output_names=["logits"], dynamic_axes = {"input_ids": {0:"batch", 1:"seq"}, "logits":{0:"batch", 1: "seq"}},
        opset_version = args.opset,
        do_constant_folding=True
    )
    print("Saved ONNX to", args.out)


if __name__ == "__main__":
    main()


## Run:
# python export_onnx.py --ckpt miniLM.pt --out miniLM.onnx
