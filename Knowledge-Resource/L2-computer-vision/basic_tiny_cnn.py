import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 🖥️ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 🧠 Model Definition
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.head(h)

# 🧮 Train loop
def run_epoch(dataloader, model, opt, criterion, scaler, train=True):
    model.train() if train else model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            opt.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)

        if train:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_samples += yb.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc

# 🚀 Main entry point — required for multiprocessing!
if __name__ == "__main__":
    # 🧼 Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    tiny_subset = Subset(train_dataset, range(256))

    # num_workers = 0 avoids multiprocessing issues on macOS
    tiny_dl = DataLoader(tiny_subset, batch_size=64, shuffle=True, num_workers=0)

    # 🧠 Model / Optim / Loss
    model = TinyCNN().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # AMP scaler — only for CUDA
    scaler = torch.amp.GradScaler(device_type="cuda") if device.type == "cuda" else None

    # 🧪 Overfit tiny set
    print("🚀 Overfitting sanity check on 256 samples...")
    for epoch in range(50):
        loss, acc = run_epoch(tiny_dl, model, opt, criterion, scaler, train=True)
        print(f"Epoch {epoch:02d} | loss: {loss:.4f} | acc: {acc:.4f}")

    print("✅ Training loop works correctly if loss ↓ and acc ↑ toward 1.0")
