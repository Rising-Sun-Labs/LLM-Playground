# lesson 2 — turn data into trainable batches
# idea: the model reads a fixed window of tokens (context window = block_size) and tries to predict the next token at each position.

import torch
from torch.utils.data import Dataset


class CharDataset:
    def __init__(self, path: str, block_size: int, device="cuda" if torch.cuda.is_available() else "cpu", train_split=0.9):
        text = open(path, "r", encoding="utf-8").read().lower()
        self.vocab = sorted(list(set(text)))
        self.stoi = {ch:i for i, ch in enumerate(self.vocab)}
        self.itos = {i:ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(train_split * len(data))
        self.train_data, self.val_data= data[:n], data[n:]
        self.block_size=block_size
        self.device = device

    def get_batch(self, split: str, batch_size: int):
        src = self.train_data if split == "train" else self.val_data
        idx = torch.randint(0, len(src)- self.block_size-1, (batch_size,))
        x = torch.stack([src[i:i+self.block_size] for i in idx])
        y = torch.stack([src[i+1:i+1+self.block_size] for i in idx])
        return x.to(self.device), y.to(self.device)

