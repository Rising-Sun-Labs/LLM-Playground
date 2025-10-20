# nn.Module (building neural nets)
# 1. define a small network and train on a toy dataset

import torch
import torch.nn as nn
from sklearn.datasets import make_moons
import numpy as np

# Define model
class MLP(nn.Module):
    def __init__(self, d_in=2, d_h=64, d_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(),
            nn.Linear(d_h, d_h), nn.ReLU(),
            nn.Linear(d_h, d_out)
        )
    def forward(self, x):
        return self.net(x)
    

# Generate toy data
X, y = make_moons(n_samples=4000, noise=0.2, random_state=0)
X = torch.from_numpy(X.astype(np.float32))
y = torch.from_numpy(y.astype(np.int64))

# Model, optimizer, loss
model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# Training loop
for step in range(1500):
    opt.zero_grad(set_to_none=True)
    logits=model(X)
    loss=crit(logits, y)
    loss.backward()
    opt.step()

    if step % 300 == 0:
        print(f"Step {step:4d} | loss = {loss.item():.4}")

print("load weights", model.state_dict()/model.load_state_dict())

# checkpoints
# you can: build an nn.Module, call it, get a loss, and step an optimizer
# try model.state_dict() / model.load_state_dict()  to save and load weights

