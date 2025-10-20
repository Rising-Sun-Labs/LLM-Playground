import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# ==============================
# 🖥 Device
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ==============================
# 🔹 Model definition (same as training)
# ==============================
import torch.nn as nn

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

# ==============================
# 📦 Test Dataset
# ==============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

# ==============================
# 🔹 Load checkpoint
# ==============================
model = TinyCNN().to(device)
checkpoint = torch.load("tinycnn_best.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

# ==============================
# 🔹 Evaluation
# ==============================
all_pred, all_true = [], []

with torch.no_grad():
    for xb, yb in test_dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        all_pred.append(preds.cpu())
        all_true.append(yb.cpu())

y_true = torch.cat(all_true).numpy()
y_pred = torch.cat(all_pred).numpy()

# ==============================
# 🔹 Confusion matrix
# ==============================
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_ds.classes)

fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("CIFAR-10 Confusion Matrix")
plt.show()

# ✅ Print overall accuracy
acc = (y_true == y_pred).mean()
print(f"Test Accuracy: {acc:.4f}")
