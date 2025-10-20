# Lesson 26: Shipping

0) Initial Reference architecture:

            [Client/Web] ──HTTPS──> [API Gateway (FastAPI)]
                                     |  ├─ AuthZ/RateLimit/PII-Redact
                                     |  ├─ Token Budgeter / Decoding Params
                                     |  ├─ Router → { vLLM | TRT-LLM | Small LM }
                                     |  └─ Tools Proxy (RAG, pytest, etc.)
                                     |
                                     ├──> [vLLM Server]  (bf16, paged KV, streaming)
                                     ├──> [TRT-LLM]      (fp16/int8, TP, chunked prefill)
                                     └──> [Small LM]     (fallback, speculative draft)
                                     |
                                     ├──> [RAG Service]
                                     |       ├─ FAISS + BM25 + Reranker
                                     |       └─ Corpus Store (Docs/PDFs, per-tenant)
                                     |
                                     ├──> [Agent Runner]
                                     |       ├─ Planner/Executor/Verifier
                                     |       └─ Sandboxed Tools (zip, pytest, mypy...)
                                     |
                                     ├──> [Observability]
                                     |       ├─ Prometheus (/metrics), OTEL traces
                                     |       └─ Logs (PII-redacted, request hashes)
                                     |
                                     └──> [Secrets/KMS]  [Redis]  [S3/Blob: ckpts, shards]

        
1) Repo Layout (single mono-repo, buildable )

            /app
              /gateway/            # FastAPI routes, auth, rate limit, budgeter, routers
              /engines/            # vLLM/TRT clients, fallbacks, streaming glue
              /rag/                # index_build, retrieve, rerank, prompt composer
              /agent/              # plan, exec, verify, patch protocol tools
              /obs/                # middleware, metrics, tracing
              /security/           # redaction, PII patterns, allowlists
              /configs/
                prod.yaml          # model ids, endpoints, budgets, SLOs
                staging.yaml
              /scripts/
                launch_vllm.sh
                launch_trt.sh
                build_index.py
              /eval/               # offline eval suite & goldens
              /infra/
                docker/            # Dockerfiles
                k8s/               # Deployments, HPAs, Ingress, Secrets refs
                terraform/         # (optional) networks, buckets, KMS, clusters

2) environment & secrets
    - Config via configs/*.yaml; load by APP_ENV (staging/prod).
    - Secrets in KMS/Secret Manager; inject as env refs in k8s (valueFrom).
    - Egress: deny by default for model pods; allowlist RAG fetchers.
    - TLS: terminate at Ingress; mTLS internal if available.

3) CI/CD Pipeline
    - CI (on PR): lint, unit tests, mini-pipeline (1% data), eval suite (lesson 11), safety tests, build Docker images.
    - Artifact: push images + tag model/dataset IDs.
    - CD: deploy to staging (blue); run smoke (latency, /metrics) + online goldens (lesson 23).
    - Canary prod 10% with auto-rollback on guardrails (latency, 5xx, safety).
    - Promote to 100% or fall back to green version.

4) Config knobs
    ```
   models:
      primary: { engine: vllm, base_url: "http://vllm:8001", max_ctx: 32768 }
      lowlat:  { engine: trt,  base_url: "http://trt:8002",  max_out: 256 }
      small:   { engine: vllm, base_url: "http://small:8001" }
   decoding: { temperature: 0.8, top_p: 0.9, rep_penalty: 1.1, stop: [] }
   routing:
    rules:
      - match: {domain: "rag_long"} -> primary
      - match: {domain: "chat_short"} -> lowlat
      - match: {uncertainty_gt: 0.5} -> primary else small
   budgets:  { max_prompt: 6000, max_new: 512 }
   rates:    { per_ip_min: 60, per_key_min: 120 }
   safety:   { pii_redact: true, citation_required_rag: true }

     ```

5) Go live checklist
    - vLLM up with correct tokenizer + --max-model-len; health OK.
    - TRT-LLM engine built with paged KV; health OK.
    - Gateway /health, /metrics, streaming SSE/WS working.
    - Redaction on inputs/outputs; logs store hashes only.
    - Rate limit + quota enforced; 429 tested.
    - RAG index built (manifest pinned); /rag_answer returns citations.
    - Agent tools sandboxed; path allowlist; timeouts enforced.
    - Dashboards: latency p50/p95, 5xx, tokens/sec, queue depth, safety fails.
    - Canary 10% with auto-rollback rules.
    - Runbooks linked (below).   

6) SLOs & alerts (minimal set)

TTFT p95 ≤ 800 ms (chat_short), ≤ 2.5 s (rag_long).

5xx rate < 1%; timeouts < 0.5%.

Online goldens pass ≥ 0.9; safety ≥ 0.95.

Alert if SLO breached for 5 min or safety fails spike.

7) runbooks (copy/paste)
7.1 Incident (latency spike / 5xx)

Check dashboards: queue depth, GPU util, engine health.

Flip router to backup engine (lowlat→primary or vice versa).

Reduce max_new and batch window; drain long prompts.

If engine unhealthy: restart pod; if repeated, rollback to last image tag.

Postmortem: attach logs (request IDs), engine metrics, change diff.

7.2 Safety regression

Auto-rollback triggers → confirm in experiment dashboard.

Lock offending route to small or base SFT.

Inspect recent prompts; add SFT refusals; re-canary.

7.3 RAG index stale/bad

Pin previous corpus manifest; evict cache.

Rebuild index offline; re-enable canary slice with online goldens.

8) cost controls (flip today)

Token budgeter trims oldest memory/retrieved chunks.

Cache deterministic prompts (hash prompt+decode settings).

Route simple prompts to small; escalate on uncertainty.

Enable KV INT8 and try INT4 weights (if quality gates pass).

Nightly utilization report (GPU hours, tokens/day by endpoint).

9) data & model registry (one page)

Dataset IDs → manifests + metrics

Model IDs → config (arch, rope, tokenizer hash), evals, compression card

Deployment records → image tags, infra commit, experiment link

Keep a tiny YAML/DB; surface in /version endpoint.

10) staged rollout plan (first week)

Day 1–2: shadow traffic, fix any perf/safety snags.

Day 3: canary 10% (chat_short), 5% (rag_long), online goldens on.

Day 4–5: widen to 50% if KPIs green; start collecting feedback→DPO pairs.

Day 6–7: 100% rollout; schedule nightly evals & privacy canaries.

11) minimal k8s (one deployment each)

gateway-deploy.yaml: FastAPI + HPA (CPU 70%, Q depth).

vllm-deploy.yaml: --gpu-memory-utilization=0.9, --max-model-len.

trt-deploy.yaml: TP size envs, chunked prefill.

rag-deploy.yaml: Reranker CPU/GPU; FAISS index mounted RO.

tools-deploy.yaml: sandboxed runner (no net), tight limits.

ingress.yaml: TLS, sticky cookies for A/B, rate limit at edge.

12) smoke tests (automate on deploy)

/generate → deterministic prompt (temp=0) → exact hash match.

/rag_answer → known doc → required citation [1].

Agent: tiny repo → pass 1 test.

Safety prompt → required refusal string.

Metrics endpoint returns counters; traces reach collector.