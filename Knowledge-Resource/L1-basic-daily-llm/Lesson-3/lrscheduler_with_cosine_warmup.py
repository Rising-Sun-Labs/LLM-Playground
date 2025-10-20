# Adding a cosine Learning Rate scheduler with warmup to your Trainer class

import torch
from transformers import get_cosine_schedule_with_warmup

class Trainer:
    def __init__(self, model, optimizer, criterion, device=None, num_warmup_steps=100, num_training_steps=1000):
        self.model=model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # cosine LR scheduler with warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def fit(self, train_loader, val_loader=None, epochs=5):
        total_steps = epochs * len(train_loader)
        step = 0

        for ep in range(epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

                # Step the LR scheduler after every optimizer step
                self.scheduler.step()
                step += 1

                if step % 100 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Step {step}/{total_steps} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

            if val_loader:
                self.evaluate(val_loader)
        print("Training Complete")

    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total, correct = 0, 0
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            pred = self.model(xb).argmax(1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
        acc = correct / total
        print(f"Validation accuracy: {acc:.4f}")
    
    def save(self, path="cnn_weights.pt"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path="cnn_weights.pt"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")
        