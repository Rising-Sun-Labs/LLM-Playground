# Practice

    - Create configs/staging.yaml & configs/prod.yaml with routing/budgets above.
    - Stand up vLLM and TRT-LLM with your exported model; verify tokenizer parity.
    - Wire the gateway router + streaming; enable redaction and /metrics.
    - Build a small RAG corpus and confirm citations.
    - Add canary (10%) + auto-rollback (latency/safety).
    - Run smoke tests and flip staging → prod canary.

Stretch

    - Multi-GPU TP=2 on TRT-LLM; benchmark TTFT/throughput. 
    - Specul ative decoding (small→primary) if your engines permit.
    - Per-tenant RAG indexes + envelope encryption.