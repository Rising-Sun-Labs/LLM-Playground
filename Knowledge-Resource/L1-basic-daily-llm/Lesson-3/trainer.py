# Training Loop class
import torch

class Trainer:
    """
    A reusable training and evaluation loop for PyTorch models
    """
    def __init__(self, model, optimizer, criterion, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.m = model.to(device)
        self.o = optimizer
        self.c = criterion
        self.device = device

    def fit(self, train_loader, val_loader=None, epochs=5):
        for ep in range(epochs):
            self.m.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.o.zero_grad()
                loss = self.c(self.m(xb), yb)
                loss.backward()
                self.o.step()
            print(f"✅ Epoch {ep+1}/{epochs} done.")
            if val_loader:
                self.evaluate(val_loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.m.eval()
        total, correct = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            pred = self.m(xb).argmax(1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
        print(f"🎯 Validation Accuracy: {correct / total:.2f}")


# Trains model for epochs
# Run validation after each epoch
# Print accuracy
