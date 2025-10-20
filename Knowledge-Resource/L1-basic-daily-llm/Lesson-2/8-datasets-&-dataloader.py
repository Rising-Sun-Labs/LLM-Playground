# Batching like a pro
import torch
import torchvision as tv
from torch.utils.data import DataLoader

if __name__ == "__main__":
    train = tv.datasets.CIFAR10(
        "./data",
        train=True,
        download=True,
        transform = tv.transforms.ToTensor()
    )

    train_loader = DataLoader(
        train,
        batch_size=128,
        shuffle=True,
        num_workers=2,          # or 0 if you're still on macOS MPS
        pin_memory=False        # MPS doesn't support pinned memory
    )
    xb, yb = next(iter(train_loader))
    print(xb.shape, yb.shape)
#  Write a small Dataset that reads images from images and label from a CSV