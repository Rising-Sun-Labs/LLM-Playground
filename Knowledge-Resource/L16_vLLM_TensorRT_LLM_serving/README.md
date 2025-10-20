# Lesson 16: vLLM/TensorRT-LLM Serving 
    - Continuous batching, paged KV cache + continuous batching and TensorRT-LLM (GPU kernels + quantization + tensor parallel.

0) Goals: High-Throughput serving (vLLM & TensorRT-LLM)
    
   - When to pick **vLLM** vs **TensorRT-LLM (TRT-LLM)**
   - Export/convert your weights.
   - Launchers (single/multi-GPU) with **continuous batching**
   - **Paged KV cache**, **tensor parallel**, **quantization**(INT8/FP8/INT4)
   - FastAPI gateway that proxies to these runtimes
   - Bench checklist & gotchas.

1) **Choose your engine**
   - vLLM
     - Pros: dead-simple to run; continuous batching, paged KV cache, OpenAI-compatible API, strong throughput for decode-bound LMs.
     - Cons: fewer low-level kernal/quantization knobs than TRT-LLM; best with standard architectures.
   
   - TensorRT-LLM
     - Pros: max perf on NVIDIA; **INT8/FP8/INT4**, kernel fusion, **tensor parallel(TP)**, multi-GPU/multi-node; great for high QPS + tight latency SLOs.
     - Cons: more build/deploy effort: conversion step.

- Rule of thump: start with **vLLM** (fast iteration), move to **TRT-LLM** when you need every last token/sec and low latency at scale.


