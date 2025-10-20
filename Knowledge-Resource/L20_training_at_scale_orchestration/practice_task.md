# Practice:
    - Elastic torchrun wrapper + NCCL envs in your launcher scripts.
    - Sharded streaming dataset with epoch-and-rank aware ordering.
    - Atomic, versioned checkpoints (latest + milestones) with RNG & data cursor.
    - SIGTERM hook for quick-save; run a kill test to verify resume.
    - Prometheus metrics for tokens/sec, queue depth, checkpoint success, restarts.
    - Divergence guard (NaN/high-loss) with auto-retry from last good + LR backoff.

Stretch

    - Support resume at different world size (e.g., 16→8 GPUs).
    - Add async eval and EMA weights for export checkpoints.
    - Build a tiny checkpoint registry (JSON) with step → path → md5 → notes.