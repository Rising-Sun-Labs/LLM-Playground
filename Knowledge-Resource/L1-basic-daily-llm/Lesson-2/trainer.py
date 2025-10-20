# A reusable training loop (+ validation)
import torch
# from tqdm import tqdm

class Trainer:
    def __init__(self, model, opt, crit, device=None, use_amp=False):
        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.m = model.to(self.device)
        self.o = opt
        self.c = crit
        self.use_amp = use_amp and (self.device in ["cuda", "mps"])
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def fit(self, train_loader, val_loader=None, epochs=5):
        for ep in range(epochs):
            self.m.train()
            total_loss = 0.0
            n_batches = 0



            for xb, yb in train_loader:
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                self.o.zero_grad(set_to_none=True)

                if self.use_amp:
                    with torch.autocast(self.device):
                        loss = self.c(self.m(xb), yb)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.o)
                    self.scaler.update()
                else:
                    loss = self.c(self.m(xb), yb)
                    loss.backward()
                    self.o.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            print(f"Epoch {ep+1}/{epochs} - train loss: {avg_loss:.4f}")

            if val_loader:
                self.evaluate(val_loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.m.eval()
        total = 0
        correct = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            pred = self.m(xb).argmax(1)
            total += yb.size(0)
            correct += (pred==yb).sum().item()
        acc = correct/total
        print(f"Validation accuracy: {acc:.4f}")
        return acc

    def save(self, path):
        torch.save(self.m.state_dict(), path)
        print(f"Model save to {path}")
    
    def load(self, path):
        self.m.load_state_dict(torch.load(path, map_location=self.device))
        self.m.to(self.device)
        print(f"Model loaded from {path}")