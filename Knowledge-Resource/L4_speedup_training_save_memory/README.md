## 🚀 Speeding Up & Optimizing Transformer Training (PyTorch 2.x)

Modern transformer models can be trained **much faster and more efficiently** using a few key PyTorch techniques.  
This guide walks through the **core optimizations** for better **speed**, **memory**, and **throughput** — especially useful for small or mid-sized models (like our mini-transformer).

---

## 🧠 0. Goals

- ✅ Add AMP (bfloat16/fp16) mixed precision safely  
- ⚡ Wrap model with `torch.compile` for graph-level speedups  
- 📦 Tune DataLoader and CUDA transfers  
- 🪄 Use gradient checkpointing to save memory  
- 📊 Quick throughput & memory benchmark  
- 🕵️ Intro to profiling with PyTorch profiler

---

## 🧪 1. AMP (Mixed Precision Training)

Mixed precision uses **lower precision (bfloat16 or fp16)** where possible, which:
- ✅ Speeds up training
- ✅ Reduces GPU memory usage
- ✅ Usually keeps accuracy

### 🔸 Example:

```
scaler = torch.cuda.amp.GradScaler(enabled=True)

for x, y in dataloader:
    x, y = x.to(device), y.to(device)
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(x)
        loss = crit(logits.view(-1, vocab_size), y.view(-1))
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
```

## ⚡ 2. torch.compile for Graph Fusion

`torch.compile` compiles your model for faster execution by fusing operations and optimizing graph execution.

### Example:
```
model = MiniTransformerLM(...).to(device)
model = torch.compile(model)

⏳ Works best with CUDA.
For CPUs or small models, speedup may be small.
```

## 📦 3. DataLoader & Transfer Tuning

Efficient data loading can remove bottlenecks:
    - Use `num_workers>0` to parallelize loading.
    - Use `pin_memory=True` for faster host -> GPU transfer
    - Prefetch & asynch copy where possible.

### Example:

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,      # parallel workers
        pin_memory=True     # speeds up host to device
    )
    
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

    
## 🪄 4. Gradient Checkpointing

For long sequences or deep models, memory usage explodes
Gradient checkpointing saves memory by:
- Not storing all activations
- Recomputing some forward ops during backward.

### Example:
    from torch.utils.checkpoint import checkpoint
    
    class Block(nn.Module):
        def __init__(self, ...):
            super().__init__()
            self.attn = ...
            self.mlp = ...
    
        def forward(self, x, rope):
            def fn(x):
                x = x + self.attn(x, rope)
                x = x + self.mlp(x)
                return x
            return checkpoint(fn, x)

    # Tradoff:
    - Much lower memory usage
    - Slightly slower (recomputes forward during backward).

## 📊 5. Quick Throughput & Memory Benchmark

You can quickly measure tokens/sec and peak memory:

### Example:

    import time
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    
    for _ in range(50):
        x, y = train.sample()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(x)
            loss = crit(logits.view(-1, vocab_size), y.view(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"⏳ Throughput: {50 * x.numel() / elapsed:.2f} tokens/sec")
    print(f"📈 Peak memory: {torch.cuda.max_memory_allocated()/1e6:.2f} MB")

## 🕵️ 6. PyTorch Profiler (Intro)

Profiler lets you find which ops are slow or memory heavy.

### Example:
    import torch.profiler

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        x, y = train.sample()
        logits = model(x)
        loss = crit(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
    
    print(prof.key_averages().table(sort_by="cuda_time_total"))

📊 Use torch.profiler.tensorboard_trace_handler to visualize in TensorBoard:
    `tensorboard --logdir=./profiler`



### Example: Full Training Loop (Fast)
    
    scaler = torch.cuda.amp.GradScaler()
    
    model = MiniTransformerLM(vocab_size).to(device)
    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    for step in range(1000):
        x, y = train.sample()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(x)
            loss = crit(logits.view(-1, vocab_size), y.view(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    
        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")



## Summary
| Technique              | Speed Gain  | Memory Saved | Notes                  |
| ---------------------- | ----------- | ------------ | ---------------------- |
| AMP (bfloat16)         | ✅ High      | ✅ Medium     | Easy to add            |
| torch.compile          | ⚡ High      | ➖            | Needs 2.0+             |
| DataLoader tuning      | ⚡ Medium    | ➖            | Reduces CPU bottleneck |
| Gradient checkpointing | ➖           | ✅ High       | For long sequences     |
| Profiling              | 🕵️ Insight | ➖            | Helps find slow ops    |
