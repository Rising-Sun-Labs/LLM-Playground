# Lesson 17: Data pipelines at scale
    - corpus curation, dedupe, filter, SFT/DPO pipelines with weights & Basics/MLflow, dataset versioning, or pause and wire vLLM/TRT-LLM

0) Data pipelines at scale (curate -> clean -> version -> train).
- Goals:
  - build a **reproducible** pipeline: ingest -> dedupe -> filter -> normalize -> split -> package
  - track **lineage & versions** (datasets are first-class artifacts)
  - add **quality gates** (toxicity/Pll/safety, format checks, leakage guards)
  - instrument with metrics & dashboards.
  - run **SFT/DPO** pipelines reliably (sharad, weights, curriculum)
  - ship a **data card** + auditing hooks
    

# what can go wrong (and fixes)

    - Silent leakage → enforce holdouts + cross-hash checks, diff reports across runs.
    - Distribution drift → compare new vs last run histograms; alert on KL divergence > τ.
    - Annotation inconsistency → compute agreement; re-train annotators; add guidelines.
    - Over-filtering (too strict toxicity/PII) → audit false positives; whitelist patterns (e.g., code “password=” in docs).
    - Sharding skew → randomize/reservoir sample before packing; verify per-shard stats.