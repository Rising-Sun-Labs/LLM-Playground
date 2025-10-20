# Lesson 18: Curriculum & scaling laws
    - How to size your model vs data + training budget, and stage your training for best returns

0) Goals
    - For next-token LMs, **quality ≈ f(parameters, tokens, compute)** with diminishing returns.
    - you get best ROI when you're **compute-optimal**: params and data sized to match your budget so neither starves the other.
    - Then you stage training (curriculum) so easy, short, and clean examples teach basics first, before long/hard and domain-specific stuff.

1) practical scaling rules (compute-optimal cheat sheet)
    Use these as planning defaults; you’ll refine with small ablations.

    - Let C = available training FLOPs (not theoretical peak).
    - Let N = non-embedding parameters (billions).
    - Let D = training tokens (billions), counted after tokenizer (including repeats).
    - Empirical sweet spot (Chinchilla-style) suggests roughly:
    - D ≈ 20 · N (tokens 10–30× params is common; pick 20× as a middle).
    - Compute roughly scales like C ∝ N · D (ignoring constants/optimizer overhead).

    Recipe to choose N and D
    - Decide your wallclock + hardware → estimate C (effective FLOPs).
    - Pick N and D so D ≈ 20N and N·D ≤ C / κ, where κ≈(6–8) accounts for forward+backward, optimizer, and parallelism overhead.
    - Adjust N to fit memory/parallelism constraints; keep D close to 20N.

    Back-of-envelope example
    - You can spend 2e22 FLOPs effectively (e.g., 8×A100 for ~14 days with decent utilization).
    - Choose N = 1B → D target ≈ 20B tokens.
    - FLOPs ≈ κ·N·D ≈ 6·(1e9)·(2e10) = 1.2e20 (fits well under 2e22, so you could go bigger).
    - Try N = 7B, D ≈ 140B → FLOPs ≈ 6·7e9·1.4e11 = 5.9e21 (still < 2e22). Good!

If you’re data-limited, reduce N so you keep D≈20N. If you’re compute-limited but have lots of data, cap D and increase N only if you can still step fast enough.

2) batch, sequence, and throughput

    - Global tokens/step = batch_size × seq_len.
    - Target 30k–200k tokens/step for mid-size models to keep kernels busy.
    - Prefer longer sequences over tiny batches once utilization holds; they train faster per token (prefill amortization), but watch memory.
    - Use grad accumulation to hit an effective batch even if per-GPU batch is small.

Rule of thumb: start with seq_len = 2k–4k, raise later during long-context adaptation.

3) optimizer, LR schedule, and warmup
    - AdamW with β1=0.9, β2=0.95–0.98, weight decay 0.05–0.1 is robust.
    - Learning rate scales sublinearly with model size. Good starting grid (max LR):
      - 1B: 3e-4
      - 3B: 2e-4
      - 7B: 1.5e-4
      - 13B: 1.0e-4

    - Schedule: linear warmup 1–2% of steps → cosine decay to 10–20% of max LR.
    - Gradient clipping: 1.0 (global norm).
    - Mixed precision: bf16 if possible; otherwise fp16 + GradScaler.

If loss spikes/NaNs: lower LR 2×, check β2 (try 0.95), add clip 0.5.

4) regularization and data mixing

Dropout 0.0–0.1 in MLP/attn (small models benefit more).

Token masking (for SFT) ensures you only learn on assistant spans.

Mixture: keep 80–90% general web/code corpus for base competence; the rest is domain (emails, reviews, logs).

Shuffle across sources, but preserve within-document order to keep long-range signals.

5) curriculum: staging your runs
Stage A — “Foundations”

Short, clean, diverse data; seq_len 1k–2k; higher LR.

Goal: fast convergence on basics; watch PPL and short-form goldens.

Stage B — “Domain infusion”

Mix in domain data (code reviews, emails, debugging).

Lower LR by ~1.5–2×; same seq or a bit longer.

Add SFT on small curated chats at the tail of the epoch.

Stage C — “Long context adaptation”

Raise seq_len (8k–32k) and add continued pretraining with RoPE scaling on long docs/logs.

Small LR (1e-5 to 5e-6), 100–300M tokens often suffice.

Stage D — “Behavior tuning”

SFT → DPO on your tasks.

Keep steps short (epochs measured in pairs, not tokens). Validate with your eval harness.

Why this order? Don’t waste long-context compute before the model knows language/programming; don’t do DPO before SFT teaches the format.

6) checkpoints & evaluation cadence

Save ema or best-val checkpoints every X tokens (e.g., every 2–5B tokens for big runs; more often for small).

Run eval suite (lesson 11) every 1–4 hours:

PPL on held-out,

goldens (behavior),

domain tasks (pass@k, email formatting),

safety refusals,

long-context (NIAH).

Watch for interference: after long-context or domain infusion, re-check earlier benchmarks.

7) ablations you should actually run (cheap, decisive)

LR sweep (max LR × {0.5, 1.0, 1.5}) for 50–100M tokens; pick the best slope.

Seq_len vs tokens/step holding compute fixed; pick higher throughput with no regression.

Data mix ±10–20% domain share; pick the one with better domain eval at same PPL.

Depth vs width at fixed params (e.g., 24×d vs 32×smaller d); many domains prefer deeper at the same N.

8) planning worksheet (fill these in)

Budget: GPUs × days × utilization → effective FLOPs C = ______

Target size: N (B params) = ______

Tokens: D ≈ 20N = ______ B tokens

Global tokens/step: ______ (batch × seq)

Steps: ≈ D · 1e9 / (tokens_per_step) = ______

Wallclock: (steps × step_time) (measure after warmup) = ______

Stages:

A: tokens = ___B, seq = ___, LR = ___

B: tokens = ___B, seq = ___, LR = ___

C: tokens = ___B, seq = ___, LR = ___

D: SFT/DPO pairs = ___ / ___, LR = ___

Eval cadence: every ___ steps, metrics: ___

9) when to stop early (don’t overtrain)

Validation PPL flattening over 2–3 evals → diminishing returns.

Cost curve: tokens/sec dropping (I/O bound) or p95 latency hitting infra limits—stop and re-stage (e.g., long-context later).

Downstream saturation: task pass rates improve <1–2% across a full stage—reconsider data mix or decode settings before burning more tokens.

10) common pitfalls & fixes

Too big model for too little data → overfit + poor generalization. Shrink N or augment D (or heavy repeat with stronger regularization).

Cranking seq_len too early → slower training, no gains. Keep Stage C separate.

One-shot DPO on weak SFT → brittle style; do SFT first, then small DPO.

Data leakage → ensure holdouts at document or repository granularity (lesson 21).

Inconsistent tokenizer changes mid-run → lock tokenizer & special IDs before any stage.

11) starter configs (copy/paste skeletons)

7B base (compute-optimal-ish)

N=7B, D=140B tokens, seq_len=2k→4k, tokens/step ≈ 131k

LR max 1.5e-4 → cosine → 2e-5, warmup 1%

Dropout 0.1, wd 0.05, β2=0.95

Stages: A=40B, B=80B, C=20B (8k seq, LR 5e-6), then SFT/DPO

1B compact

N=1B, D=20B, seq_len=2k, tokens/step ≈ 65k

LR max 3e-4 → 5e-5; dropout 0.1

Stages: A=10B, B=8B, C=2B (8k), then SFT/DPO