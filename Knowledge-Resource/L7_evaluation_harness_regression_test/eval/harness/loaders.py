import torch, json

# --- Minimal tokenizer to demonstrate; swap with your real one ---
class SimpleBPE:
    def __init__(self, obj): self.vocab=obj["vocab"]; self.rev={i:t for t,i in self.vocab.items()}
    @classmethod
    def load(cls, path): return cls(json.load(open(path,"r",encoding="utf-8")))
    def encode(self, text, add_special=False, max_len=256):
        ids=[]
        for w in text.strip().split():
            ids.append(self.vocab.get(w, self.vocab.get("<unk>",3)))
            ids.append(self.vocab.get("</w>", self.vocab.get("<unk>",3)))
        if add_special:
            ids = [self.vocab.get("<bos>",1)] + ids + [self.vocab.get("<eos>",2)]
        return ids[:max_len]
    def decode(self, ids):
        toks=[self.rev.get(int(i), "<unk>") for i in ids]; words=[]; cur=[]
        for t in toks:
            if t=="</w>": words.append("".join(cur)); cur=[]
            elif t.startswith("<"): continue
            else: cur.append(t)
        if cur: words.append("".join(cur))
        return " ".join(words)

# --- Model must expose forward(ids)->logits and generate(ids,...)->ids ---
def load_lm(ckpt_path: str, device="cpu"):
    obj = torch.load(ckpt_path, map_location=device)
    cfg = obj.get("config", {"vocab_size": 1200, "d_model":256, "n_layers":6, "n_heads":4, "d_mlp":1024, "max_len":256})
    # Define a matching model here or import from your codebase
    from eval.harness.simple_transformer import MiniTransformerLM
    m = MiniTransformerLM(**cfg)
    sd = obj["model"] if "model" in obj else obj
    m.load_state_dict(sd); m.to(device).eval()
    return m, cfg

def load_tokenizer(tokenizer_json: str):
    return SimpleBPE.load(tokenizer_json)
