import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import load_data, BatchLoader, ByteTokenizer
from model import MiniTransformerLM

# ======================
# 🖥 Device setup
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥 Using device: {device}")

# ======================
# 📜 Create tiny dataset if missing
# ======================
data_dir = "data"
data_file = os.path.join(data_dir, "tiny.txt")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(data_file):
    print(f"🚀 Creating tiny dataset at {data_file}")
    sample_text = """Once upon a time, in a small village, there lived a tiny language model.
It learned from characters, byte by byte, and tried to tell stories.
It loved to explore new sequences and predict the next byte with curiosity."""
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(sample_text)

# ======================
# 📂 Load data
# ======================
tok, tr_ids, va_ids = load_data(data_file)
train = BatchLoader(tr_ids, seq_len=128, batch_size=32, device=device)
val = BatchLoader(va_ids, seq_len=128, batch_size=32, device=device)

# ======================
# 🧠 Build model + optimizer
# ======================
model = MiniTransformerLM(vocab=tok.vocab_size, d=384, L=6, heads=6, mlp_mult=4, dropout=0.1, max_seq=512).to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# ======================
# 💾 Load checkpoint if exists
# ======================
ckpt_path = "miniLM.pt"
start_step = 0
best_val_loss = float("inf")

if os.path.exists(ckpt_path):
    print(f"🔄 Loading checkpoint {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    best_val_loss = state.get("best_val_loss", best_val_loss)
    start_step = state.get("step", 0)
    print(f"✅ Checkpoint loaded (step={start_step}, val_loss={best_val_loss:.3f})")
else:
    print("🚀 No checkpoint found. Starting training from scratch.")

# ======================
# 📈 Eval function
# ======================
@torch.no_grad()
def eval_loss(steps=20):
    model.eval()
    total = 0
    for _ in range(steps):
        x, y = val.sample()
        logits = model(x)
        loss = criterion(logits.view(-1, tok.vocab_size), y.view(-1))
        total += loss.item()
    model.train()
    return total / steps

# ======================
# 🚂 Training loop
# ======================
max_steps = 500
save_every = 50

for step in range(start_step, max_steps):
    x, y = train.sample()
    logits = model(x)
    loss = criterion(logits.view(-1, tok.vocab_size), y.view(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % save_every == 0:
        val_loss = eval_loss(steps=10)
        print(f"🪄 Step {step} | train {loss.item():.3f} | val {val_loss:.3f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "step": step,
            }, ckpt_path)
            print(f"💾 Saved checkpoint at step {step} (val_loss={val_loss:.3f})")

# ======================
# 🌟 Sampling function
# ======================
@torch.no_grad()
def sample(model, prefix, n_tokens=100, temperature=1.0):
    ids = torch.tensor([tok.encode(prefix)], dtype=torch.long, device=device)
    model.eval()
    for _ in range(n_tokens):
        idx = ids[:, -512:]
        logits = model(idx)[:, -1, :] / max(1e-6, temperature)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0].tolist())

# ======================
# ✨ Generate text
# ======================
print("\n=== Generated Text ===")
generated_text = sample(model, prefix="Once upon a time", n_tokens=200, temperature=0.9)
print(generated_text)
