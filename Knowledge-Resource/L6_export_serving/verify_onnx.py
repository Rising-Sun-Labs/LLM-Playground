# verify_onnx.py

import numpy as np, onnxruntime as ort, torch
from model_def import MiniTransformerLM


sess = ort.InferenceSession("miniLM.onnx", providers=["CPUExecutionProvider"])

obj = torch.load("miniLM.pt", map_location="cpu")
cfg = obj.get("config", {"vocab_size":1200, "d_model": 256, "n_layers": 6, "n_heads": 4, "d_mlp": 1024, "max_len":256})
model = MiniTransformerLM(**cfg)
sd = obj["model"] if "model" in obj else obj
model.load_state_dict(sd); model.eval()


x = torch.randint(0, cfg["vocab_size"], (2,7), dtype = torch.long)
with torch.no_grad():
    torch_logits = model(x).numpy()

onnx_logits = sess.run(["logits"], {"input_ids": x.numpy()})[0]
print("Torch logits:", torch_logits.shape, "ONNX logits:", onnx_logits.shape)
print("Mean abs diff (last-step slice): ", np.mean(np.abs(torch_logits[0, -1, :32] - onnx_logits[0, -1, :32])))


# Run:
# python verify_onnx.py     # you should see identical shapes and a small numeric diff
