# Lesson: 6 - export & serving (TorchScript/ONNX/FastAPI)
    

## 0) Goals

- Save a stable checkpoint
- export to TorchScript(JIT) and ONNX
- stand up a **FastAPI** service with:
  - `/health` (ready check)
  - `/generate` (text completion for your mini-transformer) from L3_mini_transformer 
  - optional streaming tokens (SSE-style)
- (bonus) quick **CPU quantization** for cheap inference. 