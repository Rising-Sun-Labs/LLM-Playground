import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 1) Dataset with ONLY ToTensor(no Normalize here)
train_raw = datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

# 2) Loader (num_workers = 0 is safest on macOS)
loader = DataLoader(train_raw, batch_size=256, shuffle=False, num_workers=0)


# 3) Compute channel-wise mean & std (numerically stable)
n = 0
mean = torch.zeros(3, dtype=torch.float64)
M2 = torch.zeros(3, dtype=torch.float64)    # running squared diffs (welford)

for imgs, _ in loader:
    # imgs: [B, C, H, W] in [0,1]
    b = imgs.size(0)
    n_new = n +b

    batch_mean = imgs.mean(dim=[0,2,3]).to(torch.float64)   # [C]
