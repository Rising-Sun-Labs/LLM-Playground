1. Pick a tiny dataset
    - Start with a single text file (e.g., tiny Shakespeare).
    - Goal: a plain string corpus you can load into memory.

2. Choose a simple tokenizer
   - For your first build, use character-level (each char → id).
   - Implement 3 functions: build_vocab(text), encode(str)->List[int], decode(List[int])->str.

3. Create train/val splits
   - convert full text -> integer ids
   - split 90% train / 10% validation.
   - Keep tensors on CPU/GPU as appropriate

4. Batching with context windows.
   - Decide block_size(e.g., 128).
   - Create a function `get_batch(split)` that returns:
     - x: tokens `t..t+block_size-1`
     - y: next tokens `t+1..t+block_size`
     - sample random starting indices per batch

5. Build the model (decoder-only transformer)
   - Embedding layers: token embedding + learnable positional embedding
   - N blocks(start with 2-4):
     - LayerNorm -> masked self-attention(causal mask) -> residual 
     - LayerNorm -> MLP(GELU) -> residual
   - LM head: linear projection to vocab_size.
   - Hyperparams to start: `n_embed=256, n_head=4, n_layer=4, dropout=0.1`.

6. Loss & optimization
    - objective: cross-entropy on next-token prediction `(logits, targets)`.
    - Optimizer: AdamW `(lr=3e-4)`, gradient clipping e.g.10
    - Train in a loop for a few thousand steps; print train loss

7. Sampling (Text generation)
    - Implement generate(context, max_new_tokens, temperature, top_k).
    - At each step: forward pass on last `block_size` tokens -> take last timestamp logits -> sample next id -> append.

8. Eval & sanity checks
   - Track validation loss every N steps - should trend down
   - Quick qualitative checks: does sampling preserve punctuation, line breaks, repeated patterns?

9. Save & load
   - Save: model weights, vocab(stoi/itos), and training config.
   - Load: rebuild model with same sizes, load state dict, reuse vocab.

10. Tight feedback loop
    - Tweak 1 thing at a time: `block_size`, `n_embed`, `n_layer`, `lr`.
    - if OOM: reduce batch size or dimensions.
    - If stuck: Lower LR, train longer, add data or reduce dropout.
    

