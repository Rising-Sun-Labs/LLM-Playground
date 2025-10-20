# Lesson 22: Multilingual & code specialization

0) Goals
    - add/expand multilingual capacity **without** wrecking English/code.
    - specialize for **coding** tasks (unit tests, tools, diffs)
    - choose tokenizer strategy (extend vs rebuild) safely
    - pick **training mixes** + adapters(LoRA) for cheap iteration
    - evaluate with solid **benchmarks** and slice dashboards.


1) tokenizer strategy (don’t brick your weights)
    A) keep tokenizer; add special tokens only

        - safest if your base already does OK on Latin scripts + common code.
        - add <image>, <audio>, tool markers, maybe <zh>, <hi> language tags.

        - Pros: zero remapping pains.
        - Cons: suboptimal tokenization for scripts like Chinese, Thai.

    B) extend tokenizer vocabulary

        - Add top N merges/pieces from new corpora (e.g., 5k–20k for Chinese/Arabic).
        - Initialize new embeddings from average of close neighbors (or small random with scaling).
        - Train a quick adapter stage to teach new embeddings while freezing most of the model.

        - Pros: preserves indices of old tokens; compatible with old checkpoints.
        - Cons: slightly larger embedding table; needs careful LR.

    C) new tokenizer entirely (rare)

        - Only if your old one is truly bad (e.g., whitespace BPE for CJK).
        - You must retrain or at least run a heavy continued pretrain; all positions shift.

        If you must, ship as a new major model (breaking change).

        For you: prefer B (extend) for multilingual; A for code (add fence tokens/language tags if missing).