import torch, torchvision as tv
from torch.utils.data import DataLoader, random_split

device = "cuda" if torch.cuda.is_available() else "cpu"

# transforms: augments for train, simple for val/test
train_tf = tv.transforms.Compose([
    tv.transforms.RandomCrop(32, padding=4),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

test_tf = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_full = tv.dataset.CIFAR10("./data", train=True, download=True, transform=train_tf)
test = tv.dataset.CIFAR10("./data", train=False, download=True, transform=test_tf)

# Small val splits from train
val_size = 5000
train_size = len(train_full) - val_size
train, val = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(0))

train_dl = DataLoader(train, batch_size=128, shuffle=True, num_workers = 4, pin_memory=True)
val_dl = DataLoader(val, batch_size=256, shuffle=False, num_workers = 4, pin_memory=True)
test_dl = DataLoader(test, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

xb, yb = next(iter(train_dl))
print("batch: ", xb.shape, yb.shape)