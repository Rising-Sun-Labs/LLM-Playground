# train_lm.py
import torch, torch.nn as nn, torch.optim as optim
from data import load_data, BatchLoader
from model import MiniTransformerLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🖥 Using device:", device)

tok, tr_ids, va_ids = load_data()

# 👇 Auto choose sequence length
max_seq = 128
seq_len = min(max_seq, max(8, len(tr_ids) - 2))  # at least 8 tokens long
print(f"📝 Using sequence length: {seq_len}")

train = BatchLoader(tr_ids, seq_len=seq_len, batch_size=32, device=device)
val   = BatchLoader(va_ids, seq_len=seq_len, batch_size=32, device=device)

model = MiniTransformerLM(
    vocab=tok.vocab_size,
    d=192,
    L=4,
    heads=4,
    mlp_mult=4,
    dropout=0.1
).to(device)

opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
crit = nn.CrossEntropyLoss()

@torch.no_grad()
def eval_loss(steps=20):
    model.eval()
    tot = 0
    for _ in range(steps):
        x, y = val.sample()
        logits = model(x)
        loss = crit(logits.view(-1, tok.vocab_size), y.view(-1))
        tot += loss.item()
    model.train()
    return tot / steps

best = float("inf")
for step in range(500):
    x, y = train.sample()
    logits = model(x)
    loss = crit(logits.view(-1, tok.vocab_size), y.view(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if step % 50 == 0:
        vl = eval_loss(10)
        print(f"step {step} train {loss.item():.3f} | val {vl:.3f}")
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict()}, "miniLM.pt")
            print("  saved checkpoint")

# checkpoint:  if training is unstable, try lower LR (1e-4), fewer heads, or disable dropout.

