import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from cosine_lr_warmup_early_stopping_amp_trainer import Trainer

# 🧪 Dummy dataset (replace with your real dataset)
X = torch.randn(5000, 20)
y = torch.randint(0, 2, (5000,))
dataset = TensorDataset(X, y)

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset, batch_size=64)

# 🧠 Simple model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 🕒 Scheduler parameters
num_epochs = 50
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)  # warmup for 10% of steps

# 🚀 Trainer
trainer = Trainer(
    model,
    optimizer,
    criterion,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    early_stopping_patience=5
)

trainer.fit(train_loader, val_loader, epochs=num_epochs)
