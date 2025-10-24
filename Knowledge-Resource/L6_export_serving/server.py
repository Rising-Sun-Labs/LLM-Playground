# FastAPI Server 
# Get  /health
# POST /generate (text completion)
# GET /generate_stream (SSE tokens, optional)


from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import torch, os, numpy as np
from fastapi.responses import StreamingResponse
from model_def import MiniTransformerLM, SimpleBPE

# Optional ONNX: import only if used.
try:
    import onnxruntime as ort
except Exception:
    ort = None

CKPT_PATH        = os.environ.get("MINILM_CKPT", "miniLM.pt")
TOKENIZER_PATH   = os.environ.get("TOKENIZER_JSON", "tokenizer.json")
BACKEND          = os.environ.get("BACKEND", "torch")   # torch or onnx
MAX_LEN          = int(os.environ.get("MAX_LEN", 256))


tok = SimpleBPE.load(TOKENIZER_PATH)

# Load backends:
if BACKEND  == "torch":
    obj = torch.load(CKPT_PATH, map_location="cpu")
    cfg = obj.get("config" ,{"vocab_size": len(tok.vocab), "d_model": 256, "n_layers": 5, "n_heads": 4, "d_mlp":1024, "max_len": MAX_LEN})