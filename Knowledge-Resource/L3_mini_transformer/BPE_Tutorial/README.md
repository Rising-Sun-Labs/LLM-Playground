# Mini Transformer — BPE Tokenization + Cosine LR Warmup/Decay

A hands-on guide to build and train a tiny GPT‑style (decoder‑only) Transformer that uses **BPE tokenization**, trains with a **cosine learning‑rate schedule with warmup**, and supports **mixed precision (AMP)** and **gradient accumulation**. You’ll also learn how to **adapt to your own dataset** (code/chats/docs) and **sample** text safely.

> If you prefer a guided notebook, use: **MiniTransformer_BPE_Tutorial.ipynb** (included alongside this README).

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Quickstart (TL;DR)](#quickstart-tldr)  
3. [1) Preparing a Corpus](#1-preparing-a-corpus)  
4. [2) Training a BPE Tokenizer](#2-training-a-bpe-tokenizer)  
   - [Hugging Face `tokenizers`](#hugging-face-tokenizers)  
   - [SentencePiece (alternative)](#sentencepiece-alternative)  
5. [3) Wiring the Tokenizer into the Mini‑Transformer](#3-wiring-the-tokenizer-into-the-mini-transformer)  
6. [4) Training with Cosine LR + Warmup, AMP, and Gradient Accumulation](#4-training-with-cosine-lr--warmup-amp-and-gradient-accumulation)  
7. [5) Sampling Text (Temperature & Top‑K)](#5-sampling-text-temperature--top-k)  
8. [Adapting to Your Own Dataset](#adapting-to-your-own-dataset)  
9. [Reproducibility Tips](#reproducibility-tips)  
10. [Troubleshooting](#troubleshooting)  
11. [FAQ](#faq)

---

## Prerequisites

- **Python** 3.10+  
- **PyTorch** (`pip install torch`)  
- **Utilities**: `tqdm` for progress bars  
- **Tokenizers**: 
  - Hugging Face `tokenizers` (`pip install tokenizers`)  
  - Optional: `sentencepiece` (`pip install sentencepiece`)  

On GPUs (recommended), mixed precision speeds things up automatically.

```bash
pip install torch tqdm tokenizers sentencepiece
```

---

## Quickstart (TL;DR)

1) **Get or create a corpus** (text/code/chats)  
2) **Train BPE tokenizer** → saves `bpe/tokenizer.json`  
3) **Encode corpus** using the tokenizer → build train/val splits  
4) **Train** mini‑Transformer with cosine LR + warmup, AMP, and gradient accumulation  
5) **Sample** with temperature/top‑k using the same tokenizer

Use the ready notebook: **MiniTransformer_BPE_Tutorial.ipynb**  
or the scripts from Lesson 6 (`train.py`, `sample.py`) after swapping in the **BPE dataset** shown below.

---

## 1) Preparing a Corpus

Your model learns from raw text. You can start with **tiny Shakespeare** or point to a folder of your own files.

- Supported: `.txt`, `.md`, and common code files (`.py, .js, .ts, .java, .go, .rs, .c, .cpp`).
- For chats, export logs to plain text files.

**Example (download tiny Shakespeare):**
```bash
mkdir -p data
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/input.txt
```

**Folder mode (own data):**
```
/my-corpus/
  notes/...
  repos/...
  chats/...
```

You’ll point the notebook or scripts to `/my-corpus` to ingest everything recursively.

---

## 2) Training a BPE Tokenizer

BPE (Byte Pair Encoding) learns a subword vocabulary so frequent chunks become single tokens. This shortens sequences and improves learning compared to character models.

### Hugging Face `tokenizers`

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

texts = [...]  # List[str] loaded from files (see notebook)
special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
tok = Tokenizer(BPE(unk_token="<unk>"))
tok.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=8000, special_tokens=special_tokens)

tok.train_from_iterator(texts, trainer)
tok.save("bpe/tokenizer.json")
```

- **`vocab_size`**: start with **4k–8k** for small corpora; **8k–16k** for mixed prose+code.
- Save `bpe/tokenizer.json` and reuse it for training & sampling.

### SentencePiece (alternative)

```python
import sentencepiece as spm

open("bpe/corpus.txt", "w", encoding="utf-8").write("\n".join(texts))
spm.SentencePieceTrainer.Train(
    input="bpe/corpus.txt",
    model_prefix="bpe/spm",
    vocab_size=8000,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3
)

# Usage:
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="bpe/spm.model")
ids = sp.encode("Hello world!", out_type=int)
text = sp.decode(ids)
```

Pick **either** HF tokenizers **or** SentencePiece and stick with it for the whole project.

---

## 3) Wiring the Tokenizer into the Mini‑Transformer

Replace your character dataset with a **BPE dataset**. It encodes full text to token IDs, concatenates into one long sequence, and slices fixed‑length blocks for next‑token prediction.

```python
import torch
from tokenizers import Tokenizer

class BPEDataset:
    def __init__(self, texts, block_size=256, device="cpu"):
        self.block_size = block_size
        self.device = device
        self.tok = Tokenizer.from_file("bpe/tokenizer.json")

        ids = []
        eos_id = self.tok.token_to_id("<eos>")
        for t in texts:
            ids += self.tok.encode(t).ids + ([eos_id] if eos_id is not None else [])
        data = torch.tensor(ids, dtype=torch.long)

        n = int(0.9 * len(data))
        self.train_data, self.val_data = data[:n], data[n:]
        self.vocab_size = self.tok.get_vocab_size()

    def get_batch(self, split: str, batch_size: int):
        src = self.train_data if split == "train" else self.val_data
        idx = torch.randint(0, len(src) - self.block_size - 1, (batch_size,))
        x = torch.stack([src[i:i+self.block_size] for i in idx])
        y = torch.stack([src[i+1:i+1+self.block_size] for i in idx])
        return x.to(self.device), y.to(self.device)
```

The **model** stays the same (decoder‑only Transformer with masked self‑attention). Initialize it with `vocab_size = BPEDataset.vocab_size`.

---

## 4) Training with Cosine LR + Warmup, AMP, and Gradient Accumulation

**Warmup** gently ramps the LR to avoid early instability. **Cosine decay** smoothly lowers LR later for a better final loss.

**Schedule (factor)**  
For step `t`, warmup `W`, total steps `T`, min factor `m=0.1`:
```
if t < W: factor = (t+1)/W
else:     factor = m + 0.5*(1-m)*(1 + cos(pi * (t-W)/(T-W)))
lr = base_lr * factor
```

**Recommended knobs (small GPU):**
- `n_layer=6, n_embed=384, n_head=6, block_size=256, dropout=0.1`
- `max_steps=3000–6000`
- `batch_size=128` (reduce on CPU), `grad_accum_steps=2`
- `warmup_steps=200`, `use_cosine=True`

**Mixed precision (AMP)** and **gradient accumulation** help you train larger configs with limited memory.

You can run all of this inside the notebook (Section 5), or use the improved `train.py` from Lesson 6.

---

## 5) Sampling Text (Temperature & Top‑K)

Use the **same tokenizer** you trained. Safe generation loop:
- **temperature**: scales logits (0.7–1.0 is safe; higher = more creative)
- **top_k**: keep only the top‑K logits (50–(vocab_size-1)); set `0` to disable

```python
@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens=300, temperature=0.9, top_k=80, device="cpu"):
    ctx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx = ctx[:, -model.block_size:]
        logits, _ = model(idx)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ctx = torch.cat([ctx, next_id], dim=1)
    return tok.decode(ctx[0].tolist())
```

**Examples:**
```python
print(generate(model, tok, "ROMEO:", temperature=0.8, top_k=80))
print(generate(model, tok, "def train_model(", temperature=0.7, top_k=100))
```

---

## Adapting to Your Own Dataset

1) Put your files under a folder (e.g., `/my-corpus`).  
2) Load recursively and build `texts = [...]`.  
3) Train BPE (e.g., `vocab_size=8000` or `16000` for code+prose).  
4) Build `BPEDataset(texts, block_size=256)`.  
5) Train with cosine warmup/decay until val loss plateaus.  
6) Sample with prompts relevant to your domain.

**Tips:**  
- For **code**, use **larger vocab** and **longer block_size (256–512)**.  
- If you see OOM, lower `batch_size` and raise `grad_accum_steps` (e.g., 1→2→4).

---

## Reproducibility Tips

- Save both the **checkpoint** and the **tokenizer** path together (e.g., `bpe_best.pt` includes `tokenizer_path`).  
- Fix random seeds if needed (`torch.manual_seed(42)`), but note that AMP and CUDA kernels can introduce nondeterminism.  
- Log `val_loss` and **perplexity** (`exp(val_loss)`), and keep the best checkpoint by val loss.

---

## Troubleshooting

- **`RuntimeError: selected index k out of range`**  
  Your `top_k` > vocab size. Lower `top_k` or set `top_k=0`.
- **CUDA OOM**  
  Lower `batch_size`, `block_size`, `n_layer`/`n_embed`, or increase `grad_accum_steps`.
- **Loss not decreasing**  
  Reduce LR (`1e-4`), extend warmup, train longer, increase data, or adjust dropout (`0.0–0.2`).
- **Garbled output**  
  Train more steps; raise vocab size; use more or cleaner domain data.

---

## FAQ

**Q: Should I use HF `tokenizers` or SentencePiece?**  
A: Both are great. HF integrates easily with Python and fast merges; SentencePiece is widely used and language‑agnostic. Pick one and stay consistent.

**Q: What vocab size should I pick?**  
A: Start with 4k–8k for small pure‑text corpora. For mixed code+text, 8k–16k helps reduce awkward splits.

**Q: How long should I train?**  
A: For tiny corpora, 3k–6k steps is enough to see structure. Larger corpora benefit from more steps and capacity.

**Q: Can I train on CPU?**  
A: Yes, just slower. Reduce `n_layer`, `n_embed`, or steps; lower `batch_size`.

---

**You’re all set!**  
- Notebook: `MiniTransformer_BPE_Tutorial.ipynb`  
- Tokenizer: `bpe/tokenizer.json`  
- Checkpoints: `bpe_best.pt` (best), `mini_gpt_last.pt` (if using scripts)



# 1) install deps
pip install torch tqdm tokenizers

# 2) TRAIN (choose one)
# (A) Train on a single file
python train_bpe.py --data_file data/input.txt --use_cosine

# (B) Train on a folder (recursively reads .txt/.md and code files)
python train_bpe.py --data_dir /path/to/my/corpus --use_cosine

# Optional knobs:
# --vocab_size 8000 --block_size 256 --n_layer 6 --n_embed 384 --batch_size 128 --grad_accum_steps 2
# (if CPU: reduce batch_size and perhaps block_size)

# 3) SAMPLE
python sample_bpe.py --ckpt bpe_best.pt --prompt "ROMEO:" --temperature 0.8 --top_k 80
python sample_bpe.py --ckpt bpe_best.pt --prompt "def train_model(" --temperature 0.7 --top_k 100
