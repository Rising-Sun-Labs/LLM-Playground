Define KPIs + guardrails for your next model; write EXPERIMENT.md.

    - Implement sticky bucketing and wire a canary flag in your gateway.
    - Add online goldens (1–5%): log pass rate per variant.
    - Build a feedback endpoint and store thumbs + reasons (hashed content).
    - Create a nightly feedback→DPO job that outputs a dated pairs shard and a quick eval report.
    - Stand up a small experiment dashboard (Prometheus + Grafana or your stack).
    - Add auto-rollback conditions to your canary controller.

Stretch

    - Try interleaving for your RAG citations (swap order from A vs B) and infer winner by clickthrough.
    - Add a bandit router for small vs big model based on domain + recent reward.