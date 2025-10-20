# use the cosine warmup in train
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import ImageCSVDataset
from model import SimpleCNN
from lrscheduler_with_cosine_warmup import Trainer


# Data 
transform = T.Compose(
    [
        T.Resize((128, 128)), 
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ]
)

train_ds = ImageCSVDataset("data/labels.csv", "data/images", transform)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)

# Model
model = SimpleCNN(num_classes=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# total steps = epochs * batches_per_epoch
epochs = 5
num_training_steps = epochs * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps)

trainer = Trainer(model, optimizer, criterion, num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)

trainer.fit(train_loader, epochs=epochs)
trainer.save("cnn_weights.pt")