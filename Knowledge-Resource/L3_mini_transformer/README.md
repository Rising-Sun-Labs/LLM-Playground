### Goal
- load a tiny text corpus
- tokenize it (byte/char for simplicity)
- build an LM with embedding -> Transformer blocks -> LM head
- train with causal mask
- sample text from your model


### Core components of a modern Transformer block
```
Token Embedding + Positional RoPE
→ Multi-Head Self-Attention (with causal mask)
→ Residual + RMSNorm
→ MLP
→ Residual + RMSNorm
```
- Token Embedding + Positional RoPE (Rotary Positional Encoding)
    - what it is
      - `Token Embedding`: Coverts each input token(word, subword, byte-pair, etc) into a vector of fixed dimension(say 4096).
      - `Positional Encoding`: Since Transformers have no recurrence or convolution, they need a way to know the order of tokens
      `Rope (Rotary Positional Embedding)` is one way to inject positional information.
    - Instead of adding a positional vector like in vanilla Transformers, RoPE rotates the query/key vectors in attention space by an angle depending on the position. This is mathematically elegant and works well for extrapolation.
  
      - **📌 Example :** Suppose our vocabulary has token `"the"`, and its embedding is 4-D vector:
      ```"The" → [1.2, -0.7, 0.3, 0.5]```
      - For position p = 0, RoPE rotates the vector minimally (angle = 0).
      - For position p = 5, RoPE applies a rotation on pairs of elements in the vector.
      - If vector = [x1, x2, x3, x4] and RoPE uses angles θ:
        ```rotated = [ x1*cos(θ1) - x2*sin(θ1), 
            x1*sin(θ1) + x2*cos(θ1),
            x3*cos(θ2) - x4*sin(θ2),
            x3*sin(θ2) + x4*cos(θ2) ]
        ```
      - This makes the embedding position-aware without adding anything.
    
- Multi-Head self-attention (with Causal Mask)
  - what it is
    - Each token looks at previous tokens to compute its representation
    - `"Multi-head"` means multiple attention mechanisms run in parallel with different learned-projections -> captures different kinds of relationships
    - `"Causal mask"` ensures a token cannot attend to future tokens (important for autoregressive model like GPT)
      - **📌 Example :**
          ```input
        ["The", "cat", "sleeps"]
        - suppose at position of "sleeps", the model attends
        - to "The" with weight 0.2
        - to "cat" with weight 0.6
        - to "sleeps" itself with weight 0.2
        ```
        - Causal mask ensures `"The"` cannot see `"cat"` and `"sleeps"` before it - at generation time, we only look backwards
        ```
         Attention(Q, K, V) = softmax( (QK^T) / √d_k + mask ) V
        
        - Q = queries
        - K = keys
        - V = values
        - mask = -inf for future tokens -> softmax -> 0
        ```
        - Multi-head means we do this multiple times and concatenate results
        ```
        head_1_output ⊕ head_2_output ⊕ ... ⊕ head_h_output
        ```
    
- Residual Connection + RMSNorm
  - what it is:
    - `Residual connection`: Add input of a sub-layer back to its output -> stabilizes training and helps gradients flow.
    - `RMSNorm(Root Mean Square LayerNorm)`: A lightweight normalization method that normalizes activations based on RMS instead of mean & variance. Faster than LayerNorm, works well in large LMs.
    ```output = RMSNorm(x + Attention(x))``` 
    - this prevents the network from "forgetting the original input signal".
    - **📌 Example :**
    ```
    input = [1.0, 2.0, 3.0]
    attention_output = [0.5, 0.5, 0.5]
    
    Then Residual:
    sum = [1.5, 2.5, 3.5]
    
    RMSNorm normalizes this vector so its RMS magnitude is fixed(e.g., scale 1).
    ```

- MLP (Feedforward Network)
  - what it is:
    - After attention, we apply a `two layer fully connected network` with a non-linear activation (e.g., SwiGLU or GELU).
    - This lets the model transform information token-wise (no mixing between tokens here -> already handled by attention).
    ```
    output = W2 * activation (W1 * x)
    
    in LLaMA:
    activation = SwiGLU(x)
    
    # Example:
    
    x = [1.5, 2.5, 3.5]
    W1 transforms it to hidden size (e.g., 4096 → 11008)
    activation adds nonlinearity
    W2 brings it back to original size (e.g., 11008 → 4096)
    
    This allows the model to build complex features out of linear combinations of attention outputs.
    ```
    
- Residual + RMSNorm again
  - After the MLP block, another residual connection and RMSNorm are applied.
  ```
  X = RMSNorm(x + MLP(x))
  this gives the final output of one Transformer layer.
  ```
  
- 📜 Full Transformer Layer (Autoregressive, like GPT)
```
x ← token embeddings + RoPE
↓
attention_out ← MultiHeadAttention(x, mask=causal)
↓
x ← x + attention_out
x ← RMSNorm(x)
↓
mlp_out ← MLP(x)
↓
x ← x + mlp_out
x ← RMSNorm(x)

Repeat this for many layers (e.g., 30-80 layers in modern models
```

## Sample Pseudocode Example
```
# x: [batch, seq_len, dim]

# Token embeddings
x = token_embedding(tokens)
x = apply_rope(x)                     # rotary positional encoding

# Attention
att_out = causal_self_attention(x)    # multi-head
x = x + att_out                       # residual
x = rms_norm(x)

# MLP
mlp_out = feedforward(x)
x = x + mlp_out                       # residual
x = rms_norm(x)

```

## Summary Table
| Component                 | Purpose                                  | How it works               |
| ------------------------- | ---------------------------------------- | -------------------------- |
| Token Embedding           | Convert tokens to vectors                | Lookup table               |
| RoPE                      | Encode positions                         | Rotates embeddings         |
| Multi-Head Self-Attention | Let each token attend to previous tokens | QK^T, softmax, causal mask |
| Residual Connection       | Stabilize training                       | Add input + output         |
| RMSNorm                   | Normalize activations                    | RMS-based normalization    |
| MLP                       | Non-linear feature transformation        | Dense → activation → Dense |



5) sanity checks & debugging
- overfit a tiny slice: take 50KB of text and confirm loss goes near 0.
- watch val loss: if it rises while train loss falls → overfitting; add dropout or shorten training.
- NaNs? lower LR, check for exploding grads (add torch.nn.utils.clip_grad_norm_).

6) optional upgrades (pick one)
- Tokenizer: replace byte tokenizer with your own BPE from earlier lessons.
- RoPE scaling: extend max context (YaRN/NTK ideas) once you’re comfy.
- FlashAttention: swap attention kernel to speed up long sequences.
- Weight tying: set self.head.weight = self.tok.weight to share embeddings.


### Steps to run
- Step 0: Setup environment: Make sure you have Python 3.10+ and PyTorch installed.
    - python -m venv venv
    - source venv/bin/activate   # macOS/Linux
    # venv\Scripts\activate    # Windows
    - pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU version
- Your folder structure should look like this:
```
L3_mini_transformer/
│
├─ data.py
├─ model.py
├─ train_lm.py
├─ sample.py
└─ (optional) data/tiny.txt
```
You don’t need tiny.txt; the script will generate a small dataset automatically if it doesn’t exist.
- Step 1: Train the model
```
Run:
python train_lm.py
```
    - What happens here:
        - Data Loading:
            data.py loads tiny.txt or generates a small dummy dataset.
            Splits into training (90%) and validation (10%).
        - Batching:
            BatchLoader randomly samples sequences for training.
        - Model Initialization:
            MiniTransformerLM creates embeddings, attention layers with RoPE, MLPs, and RMSNorm.
        - Training Loop:
            Optimizer: AdamW
            Loss: CrossEntropy
            For every step:
                Sample a batch → compute logits → compute loss → backprop → update weights
        - Checkpointing:
            - Every 50 steps, evaluates validation loss.
            - Saves model to miniLM.pt if validation improves.
- Step 2: Generate text (sampling)
  - Once training finishes or after some checkpoints:
  ```python sample.py
  ```
  - What happens here:
    - Load miniLM.pt checkpoint.
    - Encode the prefix text with ByteTokenizer.
      - For the number of tokens you want:
            - Feed the context to the model.
            - Get logits of next token.
            - Apply temperature scaling (controls randomness).
            - Sample a token from the probability distribution.
            - Append it to the sequence.
    - Decode final token IDs back to string and print.
      You can adjust temperature:
      - 0.8 → more conservative / repetitive text
      - 1.2 → more creative / random text
- Step 3 (Optional): Continuous train + generate
      - You could combine both training and sampling in a single loop: 
```
    for step in range(total_steps):
        # training batch
        # backprop
        # every N steps:
        print(sample(model, "Once upon a time", n_tokens=50, temperature=0.9))
```
This way, the model generates text while training, which is useful for monitoring learning progress.
- Step 4: Adjustments for CPU vs GPU
      - CPU training is slow, so keep d=192, seq_len=128, heads=4, L=4.
      - GPU allows higher d and L (like d=384, L=6) for better quality.