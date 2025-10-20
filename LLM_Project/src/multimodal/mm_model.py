import torch
import torch.nn as nn
from ..transformer.model import TinyGPT

class ImagePrefixLM(nn.Module):
    def __init__(self, base_lm: TinyGPT, img_encoder: nn.Module, img_seq_len=8):
        super().__init__()
        self.lm = base_lm
        self.img_encoder = img_encoder
        self.img_proj = nn.Linear(self.lm.head.in_features, self.lm.head.in_features)
        self.img_seq_len = img_seq_len

    def forward(self, img, idx):
        B,T = idx.shape
        pos = torch.arange(0,T, device=idx.device).unsqueeze(0)
        tok_emb = self.lm.tok_emb(idx) + self.lm.pos_emb(pos)
        img_seq = self.img_encoder(img)
        img_seq = self.img_proj(img_seq)
        seq_len = min(self.img_seq_len, T)
        tok_emb[:, :seq_len, :] = tok_emb[:, :seq_len, :] + img_seq[:, :seq_len, :]
        mask = torch.tril(torch.ones(T,T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        x = tok_emb
        for blk in self.lm.blocks:
            x = blk(x, mask)
        x = self.lm.ln_f(x)
        return self.lm.head(x)
