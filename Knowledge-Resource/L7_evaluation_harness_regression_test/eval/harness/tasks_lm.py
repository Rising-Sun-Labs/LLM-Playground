import torch, math
from tqdm import tqdm
from .metrics import perplexity
from .loaders import load_lm, load_tokenizer

def eval_lm_ppl(ckpt_path, tokenizer_json, text_path, max_len=256, batch_size=8, device="cpu"):
    model, cfg = load_lm(ckpt_path, device=device)
    tok = load_tokenizer(tokenizer_json)
    text = open(text_path,"r",encoding="utf-8").read()
    ids = tok.encode(text, add_special=False, max_len=max_len*1000)  # long stream
    if len(ids) < 2: return {"lm.ppl": float("inf")}
    # chunks of length L; next-token prediction
    L = max_len
    xs, ys = [], []
    for i in range(0, len(ids)-L-1, L):
        xs.append(ids[i:i+L]); ys.append(ids[i+1:i+L+1])
    device_ids = lambda arr: torch.tensor(arr, dtype=torch.long, device=device)
    ce = torch.nn.CrossEntropyLoss(reduction="sum")  # sum nll over tokens
    total_nll = 0.0; total_tokens = 0
    model.eval()
    for i in tqdm(range(0, len(xs), batch_size), desc="LM PPL"):
        xb = device_ids(xs[i:i+batch_size])
        yb = device_ids(ys[i:i+batch_size])
        with torch.no_grad():
            logits = model(xb)
            nll = ce(logits.view(-1, logits.size(-1)), yb.view(-1)).item()
        total_nll += nll; total_tokens += yb.numel()
    avg_nll = total_nll / max(total_tokens,1)
    ppl = perplexity(avg_nll)
    return {"lm.ppl": ppl}
