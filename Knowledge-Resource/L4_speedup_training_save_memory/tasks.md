9) Your tasks (do these now)
    - Add AMP and torch.compile to your lesson-7 LM and log tokens/sec & VRAM.
    - Turn on grad checkpointing and bump context length from 256 → 512 (or 1024 if fits).
    - Run the profiler for 50 steps and paste the top 5 slow CUDA ops (I’ll help interpret).