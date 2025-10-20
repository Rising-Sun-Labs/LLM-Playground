# data.py
import torch, os

# Byte-level tokenizer (simple & lossless)
class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
    def encode(self, s: str):
        return list(s.encode("utf-8"))
    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="ignore")

# Load data from file or generate a tiny dataset
def load_data(path="data/tiny.txt", split=0.9):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # generate small dummy text
        tiny_text = "Hello world. This is a tiny dataset for testing the mini-transformer. " * 50
        with open(path, "w", encoding="utf-8") as f:
            f.write(tiny_text)

    text = open(path, "r", encoding="utf-8").read()
    tok = ByteTokenizer()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(ids) * split)
    return tok, ids[:n], ids[n:]

class BatchLoader:
    def __init__(self, ids, seq_len=128, batch_size=32, device="cpu"):
        self.ids = ids
        self.L = seq_len
        self.B = batch_size
        self.device = device

    # def sample(self):
    #     if len(self.ids) <= self.L:
    #         raise ValueError(f"Sequence length {self.L} is too long for dataset of size {len(self.ids)}")
    #     ix = torch.randint(0, len(self.ids) - self.L - 1, (self.B,))
    #     x = torch.stack([self.ids[i:i+self.L] for i in ix])
    #     y = torch.stack([self.ids[i+1:i+self.L+1] for i in ix])
    #     return x.to(self.device), y.to(self.device)
    #

    def sample(self):
        # 👇 Make sure seq_len fits dataset
        if len(self.ids) <= self.L + 1:
            self.L = max(2, len(self.ids) - 2)

        ix = torch.randint(0, len(self.ids) - self.L - 1, (self.B,))
        x = torch.stack([self.ids[i:i + self.L] for i in ix])
        y = torch.stack([self.ids[i + 1:i + self.L + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

# Checkpoint you understand that x are inputs and y are the next bytes
