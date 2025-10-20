import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import ImageCSVDataset
from trainer import Trainer     # file name of reusable 


# 1. Transform & Dataset
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

train_ds = ImageCSVDataset("data/labels.csv", "data/images", transform)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)

# 2. Model
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, num_classes)
        )
    def forward(self, x): return self.net(x)

model = SmallCNN(num_classes=2)

# 3. Optimizer & Loss
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# 4. Train
trainer = Trainer(model, opt, crit)
trainer.fit(train_loader, epochs=5)

# 5. Save model
trainer.save("cnn_weights.pt")

# Tips:
# 1. Set num_workers = 0 on macOS to avoid multiprocessing issues
# 2. Trainer works with any model (MLP, CNN, Transformer...).
# 3. You can reuse it for any dataset + dataloader combo
