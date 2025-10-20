# sample.py
import torch
from model import MiniTransformerLM
from data import ByteTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🧪 Using device:", device)

# Load tokenizer
tok = ByteTokenizer()

# Model config must match training!
model = MiniTransformerLM(
    vocab=tok.vocab_size,
    d=192,
    L=4,
    heads=4,
    mlp_mult=4,
    dropout=0.0  # no dropout for inference
).to(device)

# Load checkpoint
ckpt = torch.load("miniLM.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()
print("✅ Model loaded.")

# Sampling function
@torch.no_grad()
def generate(prompt, max_new_tokens=200):
    # Encode the prompt
    ids = torch.tensor(tok.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(ids[:, -128:])  # truncate context if longer than seq_len
        logits = logits[:, -1, :]      # last token logits
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0].tolist())

# Test
prompt = "Hello"
out = generate(prompt, max_new_tokens=100)
print("📝 Prompt:", prompt)
print("📜 Generated:\n", out)

# experiments:
# try temperature = 0.8 (more conservative) vs 1.2(more creative).
# swith to top-k (keep top 50 logits before softmax) or top-p nucleus sampling
