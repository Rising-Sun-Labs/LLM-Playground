# Lesson 20: Training stack bulletproof & scalable: elastic launches, streaming data, sharded checkpoints, preemptions, and auto-resume - so a node can die and your job kees cruising.

0) Goal:
   - **Elastic** multi-node launches (join/leave, auto-resume).
   - **Steaming** input pipeline (sharded datasets; backpressure)
   - **Failure-tolerant** checkpoints (atomic, versioned, resumable)
   - **Spot/preemptible** friendly (cheap, compute without tears)
   - **Metrics & health** hooks (progress, stalls, divergence)
   - Repro knobs (seeds, determinism where it counts)