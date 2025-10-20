# Train the model

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import ImageCSVDataset
from model import SimpleCNN
from trainer import Trainer

# Load dataset
train_dataset = ImageCSVDataset(
    csv_file="data/labels.csv",
    img_dir="data/images"
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Model, optimizer, loss
model = SimpleCNN(num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trainer = Trainer(model, optimizer, criterion)
trainer.fit(train_loader, epochs=5)

# Save trained model
torch.save(model.state_dict(), "cnn_weights.pt")
print("✅ Training complete. Model saved to cnn_weights.pt")


# Loads your dataset & dataloader
# Builds the model
# Trains for 5 epochs
# Save weights
