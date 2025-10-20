import torch

# Creation
a = torch.tensor([[1., 2.], [3., 4.]])  # explicit
b = torch.arange(6).reshape(2, 3)       # 0..5
c = torch.randn(3,3)                    # ~N(0,1)

# dtype & device
print(a.dtype, a.device)                # torch.float32 cpu
a_cuda = a.to("cuda") if torch.cuda.is_available() else a


# shapes & reshaping
x = torch.arange(12)                    # [0..11]
x2 = x.view(3,4)                        # shares memory; must be contiguous
x3 = x2.permute(1,0).contiguous().view(6,2)

# broadcasting
v = torch.tensor([10., 20., 30., 40.])  # [4]
y = x2.float()+v                        # [3,4] + [4] -> [3,4]

print("x2:\n", x2)
print("y:\n", y)

# Checkpoints: 
# understand: shape, dtype, device, view vs permute, broadcasting. 
# mini-exercise 
# Create a 3-D tensor of shape [2,3,4] swap axes to [3,2,4]. then flatten to [6,4].
