### practice:

- Implement **NTK scaling** for RoPE and re-export your LM with `max_sea=64k`.
- Add **sliding-window mask** for inference with `W=4096` and test latency vs full attention.
- Build a **NIAH** script for 8k/16k/32k/64k; plot EM vs needle position.
- Add **retrieval** to your server: chunk a long doc, search, and compose a prompt with citations.
- Add **conversation memory card** (summarizer) and inject it for new turns; verify it stays under 120 tokens.

Stretch:
- Train a brief **continued pretrain** on long sequence (8k->16k), then re-run NIAH.
- Adda a **global tokens** mechanism (e.g., prepend a small set per 1k tokens) in top layers. 

