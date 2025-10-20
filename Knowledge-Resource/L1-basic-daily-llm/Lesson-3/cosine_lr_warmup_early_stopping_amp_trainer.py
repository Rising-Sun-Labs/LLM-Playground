import torch
from torch.optim.lr_scheduler import LambdaLR
import math

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        num_warmup_steps,
        num_training_steps,
        early_stopping_patience=5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.best_val_loss = float("inf")
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0

        # 📈 LR Scheduler with Warmup + Cosine Decay
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(optimizer, lr_lambda)

        # ⚡ AMP GradScaler — auto handle old/new PyTorch
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
            except TypeError:
                self.scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    def fit(self, train_loader, val_loader=None, epochs=10):
        total_steps = 0
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)

                # ⚡ Forward pass with autocast
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    ctx = torch.amp.autocast("cuda", enabled=(self.device == "cuda"))
                else:
                    ctx = torch.cuda.amp.autocast(enabled=(self.device == "cuda"))

                with ctx:
                    logits = self.model(xb)
                    loss = self.criterion(logits, yb)

                # 🪜 Backward with scaled gradients
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                running_loss += loss.item()
                total_steps += 1

            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] - Train loss: {avg_train_loss:.4f}")

            # 📊 Validation + Early stopping
            if val_loader:
                val_loss = self.evaluate(val_loader)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    torch.save(self.model.state_dict(), "best_model.pt")
                    print(f"✅ New best val loss: {val_loss:.4f} — Model saved.")
                else:
                    self.early_stopping_counter += 1
                    print(f"⚠️ Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")

                    if self.early_stopping_counter >= self.early_stopping_patience:
                        print("🛑 Early stopping triggered.")
                        self.model.load_state_dict(torch.load("best_model.pt"))
                        return

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)

            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                ctx = torch.amp.autocast("cuda", enabled=(self.device == "cuda"))
            else:
                ctx = torch.cuda.amp.autocast(enabled=(self.device == "cuda"))

            with ctx:
                logits = self.model(xb)
                loss = self.criterion(logits, yb)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += yb.size(0)

        avg_loss = total_loss / len(loader)
        acc = total_correct / total_samples
        print(f"Validation Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}%")
        return avg_loss
