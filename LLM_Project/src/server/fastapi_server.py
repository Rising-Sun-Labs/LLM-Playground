from fastapi import FastAPI
from pydantic import BaseModel
import torch
from ..tokenizer.simple_bpe import SimpleBPE
from ..transformer.model import TinyGPT

app = FastAPI(title="Local LLM Server")

class GenerateIn(BaseModel):
    prompt: str
    max_new: int = 120
    temperature: float = 0.9
    top_k: int = 40

tok = None
model = None
max_len = 256

@app.on_event("startup")
def load():
    global tok, model, max_len
    tok = SimpleBPE.load("data/tokenizer.json")
    obj = torch.load("runs/lm_demo/best.pt", map_location="cpu")
    vocab_size = obj.get("vocab_size", len(tok.vocab))
    max_len = obj.get("max_len", 256)
    model = TinyGPT(vocab_size=vocab_size, max_len=max_len)
    model.load_state_dict(obj["model"] if "model" in obj else obj)
    model.eval()

@app.post("/generate")
def generate(body: GenerateIn):
    ids = tok.encode(body.prompt, add_special_bos_eos=True, max_len=max_len)
    x = torch.tensor([ids], dtype=torch.long)
    y = model.generate(x, max_new_tokens=body.max_new, temperature=body.temperature, top_k=body.top_k)
    return {"text": tok.decode(y[0].tolist())}
