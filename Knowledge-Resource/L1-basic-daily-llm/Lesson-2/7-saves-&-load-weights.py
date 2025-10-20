# nn.Module (building neural nets)
# 1. Define a small network and train on a toy dataset

import torch
import torch.nn as nn
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt


# -----
# Device configurations (GPU or CPU)
# -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------
# Define Model
# ------
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
    
# ------
# Generate toy data
# ------
X,y = make_moons(n_samples=4000, noise=0.2, random_state=0)
X = torch.from_numpy(X.astype(np.float32)).to(device)
y = torch.from_numpy(y.astype(np.int64)).to(device)

# ------
# Model, optimizer, loss
# ------
model = MLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# ------
# Training loop
# ------
for step in range(1500):
    opt.zero_grad(set_to_none=True)
    logits = model(X)
    loss = crit(logits, y)
    loss.backward()
    opt.step()

    if step % 300 == 0:
        print(f"Step {step:4d} | loss = {loss.item():.4f}")


# -----
# Save weights
# -----
torch.save(model.state_dict(), "mlp_weights.pt")
print("\n✅ Weights saved to 'mlp_weights.pt'")


# -----
# Load weights into a new model
# -----
model2 = MLP().to(device)
model2.load_state_dict(torch.load("mlp_weights.pt", map_location=device))
model2.eval()
print("✅ Weights loaded successfully.")

# -----
# Check loaded weights match
# -----
for (name1, param1), (name2, param2) in zip(model.state_dict().items(), model2.state_dict().items()):
    print(f"{name1}: equal = {torch.allclose(param1, param2)}")


# -----
# Plot decision boundry
# -----
def plot_decision_boundry(model, X_cpu, y_cpu):
    x_min, x_max = X_cpu[:, 0].min() - 0.5, X_cpu[:, 0].max()+0.5
    y_min, y_max = X_cpu[:, 0].min() - 0.5, X_cpu[:, 0].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_torch = torch.from_numpy(grid.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(grid_torch)
        preds = logits.argmax(dim=1).cpu().numpy()
    preds = preds.reshape(xx.shape)

    plt.contourf(xx, yy, preds, alpha=0.3, cmap = plt.cm.coolwarm)
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=15, cmap=plt.cm.coolwarm)
    plt.scatter(X_cpu[:, 0], X_cpu[:, 1], c=y_cpu, s=15, cmap=plt.cm.coolwarm)
    plt.title("Decision Boundry (MLP)")
    plt.show()

# plot_decision_boundry(model2, X.numpy(), y.numpy())
# Move data back to CPU for plotting
plot_decision_boundry(model2, X.cpu().numpy(), y.cpu().numpy())

# What this does:
# 1. Builds a simple 2-layer MLP with nn.Sequential
# 2. Generates make_moons data
# 3. Trains for 1500 steps with Adam
# 4. Saves weights  -> mlp_weights.pt
# 5. Loads them into a fresh model and verifies they're identical
# 6. Visualizes the learned decision boundry
