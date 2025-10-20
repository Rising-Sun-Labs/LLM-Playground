# Practice:

- Fill the planning worksheet with your real hardware and time budget.
- Pick N and D using D≈20N and check they fit your FLOPs and memory.
- Draft your Stage A–D plan (tokens, seq, LR) and paste it here—I’ll sanity-check.
- Queue three ablations: (a) LR, (b) seq_len vs throughput, (c) domain mix.
- Wire your trainer to log tokens/step, utilization, and evals at the cadence you set.

Stretch

- Try a deeper-narrower vs shallower-wider model at same params for your domain; choose the winner by your evals + speed.
- Prototype a small long-context head (top K layers use sliding window) and validate latency/quality trade-off before Stage C.