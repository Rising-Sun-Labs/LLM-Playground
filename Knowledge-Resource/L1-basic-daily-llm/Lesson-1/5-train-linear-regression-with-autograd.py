# Autograd = > automatic differentiation
import torch

torch.manual_seed(0)

# data
N = 1024
x = torch.randn(N, 1)
y = 3*x + 2 + 0.1*torch.randn(N,1)

# parameters to learn
w = torch.randn(1,1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
opt = torch.optim.SGD([w,b], lr=0.1)

for step in range(200):
    opt.zero_grad(set_to_none=True)
    yhat = x @ w + b
    loss = torch.mean((yhat - y)**2)        # MSE
    loss.backward()
    opt.step()
    if step % 40 == 0:
        print(step, "loss=", loss.item(), "w=", w.item(), "b=", b.item())


# mini-exercise
# 1. switch to Adam and compare convergence speed.

# concepts
# 1. leaf vs non-leaf tensors, .detach() to stop gradients
# avoid in-place ops on tensors that require grad (can break the graph)
