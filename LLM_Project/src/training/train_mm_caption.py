import argparse
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..tokenizer.simple_bpe import SimpleBPE
from ..transformer.model import TinyGPT
from ..multimodal.vision_encoder import TinyCNNEncoder
from ..multimodal.mm_model import ImagePrefixLM
from .data import ImageCaptionDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--save", default="runs/mm_demo")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    save = Path(args.save); save.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=save.as_posix())

    tok = SimpleBPE.load(args.tokenizer)
    ds = ImageCaptionDataset(args.captions, args.images_root, tok, img_seq_len=args.seq, max_len=args.maxlen)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    lm = TinyGPT(len(tok.vocab), d_model=256, n_layers=6, n_heads=4, d_mlp=1024, max_len=args.seq+args.maxlen).to(args.device)
    enc = TinyCNNEncoder(out_dim=256, seq_len=args.seq).to(args.device)
    model = ImagePrefixLM(lm, enc, img_seq_len=args.seq).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(ignore_index=tok.vocab["<pad>"])

    best = float("inf"); step=0
    while step < args.steps:
        for img,x,y in dl:
            img=img.to(args.device); x=x.to(args.device); y=y.to(args.device)
            logits = model(img, x)
            loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            writer.add_scalar("train/loss", loss.item(), step)
            if loss.item() < best:
                best = loss.item(); torch.save({"model": model.state_dict()}, save/"best.pt")
            if step % 50 == 0: print(f"step {step} loss {loss.item():.3f}")
            if step >= args.steps: break
    writer.close(); print(f"Done. Best {best:.3f} saved to {save}/best.pt")

if __name__ == "__main__":
    main()
