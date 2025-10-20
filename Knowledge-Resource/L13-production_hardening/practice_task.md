# Practice:
- Add **RequestContext** middleware + `/metrics` endpoint; verify Prometheus sees counters.
- Implement **rate limiting (redis)** and return `429` when exceeded.
- Add **timeouts** around generation and **circuit breaker** with a 60s window.
- Turn on **micro-batching(10ms window)** and log tokens/sec vs baseline.
- Set up a canary route for `miniLM_dpo` vs `miniLM_sft`; compare p95 latency + goldens pass rate.
- Add a **token budgeter** and log trims for oversize prompts.

Stretch: Ship OpenTelemetry traces to Jaeger and screenshot the end-to-end span waterfall.

