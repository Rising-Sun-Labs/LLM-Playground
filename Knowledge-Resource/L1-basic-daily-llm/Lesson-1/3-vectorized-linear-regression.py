import torch
torch.manual_seed(0)

# y = 3x+2+noise
N = 1024
x = torch.randn(N,1)
y = 3*x+2+0.1*torch.randn(N,1)

# solve with least squares: [w, b]
X = torch.cat([x, torch.ones_like(x)], dim=1)   # [N,2]
wb = torch.linalg.lstsq(X, y).solution          # [2,1]
print("estimated w, b:", wb.squeeze().tolist())


# What you learned: building blocks + vectorization
# Home work.
# 1. Compute mean absolute error between predictions and y
# 2. generate x with shape [N, 2] and fit y = 2*x1 - 4*x2+1 using the same trick.
