import torch
import torch.nn as nn

class TinyCNNEncoder(nn.Module):
    def __init__(self, out_dim=256, seq_len=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.seq_len = seq_len
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        h = self.conv(x)
        pooled = h.mean(dim=[2,3])
        seq = self.proj(pooled).unsqueeze(1).repeat(1, self.seq_len, 1)
        return seq
