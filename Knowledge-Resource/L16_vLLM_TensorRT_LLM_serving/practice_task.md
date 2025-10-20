# Practice tasks (do now)

0) vLLM path

    - Package your model into an HF-style folder.
    - Launch vLLM with --max-model-len you need.
    - Hit /v1/chat/completions from your existing FastAPI gateway; enable streaming.

1) TRT-LLM path

    - Convert a copy of your weights; build an engine with paged_kv_cache.
    - Turn on KV INT8 and measure VRAM vs baseline.
    - Run the TRT-LLM server and add a fallback route in your gateway.

2) Bench

    - Run the 3 workloads; collect TTFT/TPOT/throughput + GPU util.
    - Decide default engine per endpoint (chat vs RAG long-ctx).


Stretch

    - Enable tensor parallel = 2 on dual-GPU; confirm scaling.
    - Try INT4 AWQ; compare eval pass rates vs fp16. Add speculative decoding in the gateway with vLLM as target and your small LM as draft (if the engine supports it or via hybrid flow).


recommended defaults

    - General chat: vLLM, bf16, --max-model-len=32k, batch window 10ms
    - Latency-critical: TRT-LLM fp16, TP=2, chunked prefill on, max output 256
    - Long-ctx RAG: vLLM, KV paged, prompt budgeter 6–8k, streaming on
    - Cost-saver: TRT-LLM with KV INT8, weight INT8 or INT4 AWQ (validate quality)