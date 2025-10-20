import torch

x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x+1
print("y:", y)
y.backward()

print("dy/dx at x=3:", x.grad.item())       # should be 2x+2    =>  8

# requires_grad = True  tells PyTorch to track ops for backprop
# .backward() computes gradients to leaves
# .grad holds the result on leaf tensors
