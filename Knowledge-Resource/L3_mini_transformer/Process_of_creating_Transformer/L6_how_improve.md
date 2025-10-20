- lesson 6 — how to improve (one knob at a time)
    - more steps: --max_steps 5000 (will look much better)
    - bigger context: --block_size 256 (needs more memory)
    - more layers/width: --n_layer 6 --n_embed 384 (better capacity)
    - regularize: try --dropout 0.0 to 0.2, watch val loss
    - sampling: play with --temperature 0.7..1.2, --top_k 50..500

- 0) make a baseline (so you know if tweaks helps)
       ```
       python L4_training_loop.py --device cpu --max_steps 1000 --block_size 128 --n_layer 4 --n_embed 256 --dropout 0.1
    ```
       - Write down:
         - final train_loss
         - val_loss (from printed eval)
         - derived perplexity = exp (val_loss) (lower is better)
         - a short sample `L5_sample.py`
       Note: From now on, change one thing at a time and compare to baseline.
  
- 1) More steps (easy win)
    - just train longer
        ```
      python L4_training_loop.py --device cpu --max_steps 5000
      ```
      - Expect smoother text and lower val loss
      - If loss plateaus early, you'll get diminishing returns; we'll add a scheduler next.

- 2) bigger context window(`block_size`)
    - Bumps how far the model can "look back"
        ```
      python L4_training_loop.py --block_size 256 --max_steps 5000
      ```
      - If you see OOM, drop `--batch_size` (e.g., 64->32->16)
      - Bigger `block_size` helps structure (rhymes, paragraph/line patterns).

- 3) more capacity(layers/width)
    - Try a modest bump:
    ```
    python L4_training_loop.py --n_layer 6 --n_embed 384 --max_steps 5000
  ```
    - if memory is tight: keep `n_head=4`, lower batch size, or revert `block_size=128`.

- 4) regularization (dropout sweep)
    - Run a tiny sweep, keep the best val_loss:
  ```
  # Lower regularization (may overfit but can learn faster)
  python L4_training_loop.py --dropout 0.0 --max_steps 5000
  
  # default
  python L4_training_loop.py --dropout 0.1 --max_steps 5000
  
  # stronger regularization
  python L4_training_loop.py --dropout 0.2 --max_steps 5000
    ```
  - pick the checkpoint with the lowest val_loss

- 5) Sampling Knobs(for nicer text)
    - Safe ranges:
      - `--temperature`: 0.7-1.0 (lower = more conservative, fewer mistakes)
      - `--top_k`: 50-(vocab_size-1). If you use char-level, your vocab is ~60-100.
      ```
      Example:
      python L5_sample.py --temperature 0.8 --top_k 50 --max_new_tokens 400
      ```
      - If you ever see `topk index out of range`, set a smaller `--top_k` or `--top_k 0`

- 6) three ready-to-run “profiles”
   - Pick one based on your machine
     - 1. CPU-Safe 
     ```
     python L4_training_loop.py --device cpu --max_steps 5000 --batch_size 64 --block_size 128 \
     --n_layer 4 --n_embed 256 --dropout 0.1
     ```
    - 2. Mid GPU (Balanced quality)
      ```
      python L4_training_loop.py --device cuda --max_steps 5000 --batch_size 128 --block_size 256 \
      --n_layer 6 --n_embed 384 --dropout 0.1
      ```
    - 3. Memory-tight GPU
      ```
      python L4_training_loop.py cuda --max_steps 5000 --batch_size 32 --block_size 128 \
      --n_layer 6 --n_embed 384 --dropout 0.1
      ```

  - 7) Small but powerful code upgrades
      - These make all the above knobs safer/easier
        - vii.i) Warmup + Cosine LR (Stability + better final loss)
          - patch your L4_training_loop.py: add these args
              ```
              # add the argparse in L4_training_loop.py
              p.add_argument("--warmup_steps", type=int, default=200)
              p.add_argument("--use_cosine", action="store_true")
              ```
          - Add a scheduler after optimizer:
              ```
              opt = optim.AdamW(model.parameters(), lr=args.lr)
          
              def lr_schedule(step):
                  # linear warmup
                  if step < args.warmup_steps:
                      return max(1e-8, (step+1) / max(1, args.warmup_steps))
                  if not args.use_cosine:
                      return 1.0
                  # cosine decay to 10% of base LR
                  progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
                  min_factor = 0.1
                  return min_factor + 0.5 * (1-min_factor) * (1+math.cos(math.pi * progress))
              ```
          - Apply per step inside training loop (before `opt.step()`):
              ```
              for step in trange(args.max_steps, desc="training"):
                  # ... forward/backward ...
                  # set per-step LR
                  for pg in opt.param_groups:
                      pg["lr"] = args.lr * lr_schedule(step)
                  opt.step()

              Now run:
            python L4_training_loop.py --max_steps 5000 --use_cosine --warmup_steps 200
              ```
        - vii.ii) Mixed precision (faster on CUDA, same quality)
          -  Only if you have GPU. In `L4_training.py`, near the top:
              ```
              from torch.cuda.amp import autocast, GradScaler
              scaler = GradScaler(enabled=("cuda" in str(device)))
            
              # Replace the forward/backward part of your loop:
             opt.zero_grad(set_to_none=True)
             with autocast(enabled=("cuda" in str(device))):
                _, loss = model(x, y)

             scaler.scale(loss).backward()
             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
             scaler.step(opt)
             scaler.update()

             # Note: (And remove the old loss.backward(); opt.step() lines.)
              ```
        - vii.iii) Gradiant accumulation (acts like bigger batch)
            - Add an arg:
                `p.add_argument("--grad_accum_steps", type=int, default=1)`
            - Change the loop:
                ```
                opt.zero_grad(set_to_none=True)
                for micro in range(args.grad_accum_steps):
                    x, y = ds.get_batch("train", args.batch_size // args.grad_accum_steps or 1)
                    with autocast(enabled=("cuda" in str(device))):
                        _, loss = model(x, y)
                        loss = loss / args.grad_accum_steps
                    scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                ```
            - Now run:
                `python L4_training_loop.py --device cuda --batch_size 128 --grad_accum_steps 4`
                `NOTE:` (effective batch ≈ 512 without exploding memory)
        - vii.iv) Save the best checkpoint (by val loss)
            - Add args `p.add_argument("--save_best", type=str, default="mini_gpt_best.pt")`
            - Track best inside training:
                ```
                   best_vl = float("inf")
                   if (step+1) % args.val_every == 0:
                       vl = eval_loss("val", iters=20)
                       if vl < best_vl:
                           best_vl = vl
                           torch.save({...}, args.save_best)
                           print(f"new best! val_loss={vl:.3f} -> saved {args.save_best}")
                ```
        - viii) Reading results (what "good" looks like)
            - Val loss steadily drops early, then flattens.
            - Perplexity falls (e.g., 20->10->7 for a small char model on tiny shakespeare as you scale steps/capacity).
            - Samples start to mimic punctuation, line breaks, word-like clusters.
            **If val loss goes up when you add layers/width:**
            - You're overfitting -> increases dropout, or add steps with scheduler, or use bigger dataset.
        
        - ix) Sampling presets you can try
            ```
                # conservative, coherent
                python L5_sample.py --temperature 0.7 --top_k 50

                # balanced
                python L5_sample.py --temperature 0.9 --top_k 100

                # spicier/creative
                python L5_sample.py --temperature 1.1 --top_k 0
            ```
        - x) a safe "next level" recipe (copy/paste)
          - Good mix of all knobs (GPU assumed; reduce if CPU):
            ```
            python L4_training_loop.py \
            --device cuda \
            --max_steps 6000 \
            --batch_size 128 \
            --grad_accum_steps 2 \
            --block_size 256 \
            --n_layer 6 --n_embed 384 --n_head 4 \
            --dropout 0.1 \
            --use_cosine --warmup_steps 200 \
            --val_every 200
            
            # Then sample:
            python L5_sample.py --device cuda --temperature 0.8 --top_k 80 --max_new_tokens 500
            ```
  

### How to use Lesson 6
- Baseline (CPU-safe):
    ```
  python train.py --device cpu --max_steps 2000 --batch_size 64 --block_size 128 \
  --n_layer 4 --n_embed 256 --dropout 0.1 --val_every 200
    ```
  
- Better quality (GPU if you have)
    ```
    python train.py --device cuda --max_steps 5000 --batch_size 128 --block_size 256 \
  --n_layer 6 --n_embed 384 --dropout 0.1 --val_every 200 \
  --use_cosine --warmup_steps 200
    ```

- Memory tight (but strong):
    ```
    python train.py --device cuda --max_steps 6000 --batch_size 64 --grad_accum_steps 2 \
  --block_size 256 --n_layer 6 --n_embed 384 --dropout 0.1 \
  --val_every 200 --use_cosine --warmup_steps 200
    ```
-- Sample (safe defaults)
    ```
    python sample.py --device cpu --temperature 0.8 --top_k 50 --max_new_tokens 400
    # or load the last checkpoint:
    python sample.py --ckpt mini_gpt_last.pt --temperature 0.9 --top_k 80
    ```