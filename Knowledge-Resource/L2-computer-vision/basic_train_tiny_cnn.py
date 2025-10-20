# basic_train_tiny_cnn.py
import math
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# --- AMP imports chosen by device/compatibility ---
# We prefer torch.cuda.amp for CUDA (GradScaler exists there).
if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler
    AMP_ON_CUDA = True
else:
    # For non-CUDA, autocast exists but GradScaler isn't needed/usable.
    try:
        # new API fallback (no GradScaler on CPU/MPS)
        from torch.amp import autocast  # noqa
    except Exception:
        # ultimate fallback
        from torch.cuda.amp import autocast  # noqa
    GradScaler = None
    AMP_ON_CUDA = False

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print("Device:", device)

# -------------------------
# Model
# -------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_p=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        h = self.dropout(h)
        return self.head(h)

# -------------------------
# Data + Augmentations
# -------------------------
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.2)
])

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

# Download / prepare
train_full = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_tf)

# split validation
val_size = 5000
train_size = len(train_full) - val_size
train_set, val_set = random_split(train_full, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42))

# For MPS avoid pin_memory and set num_workers = 0
num_workers = 0 if device.type == "mps" or device.type == "cpu" else 4
pin_memory = True if device.type == "cuda" else False

train_dl = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_dl = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_dl = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# tiny overfit subset loader (sanity check)
tiny_dl = DataLoader(Subset(train_set, range(256)), batch_size=64, shuffle=True, num_workers=0)

# -------------------------
# Cosine LR with manual warmup helper
# -------------------------
def make_cosine_with_warmup(optimizer, warmup_steps, total_steps, eta_min=1e-5):
    # We'll use CosineAnnealingLR for decay portion and manual warmup before it.
    decay_steps = max(total_steps - warmup_steps, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=eta_min)
    return scheduler

# -------------------------
# Training / evaluation utils
# -------------------------
def train_one_epoch(model, dl, optimizer, criterion, scaler, epoch, global_step, warmup_steps, base_lr):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    pbar = tqdm(dl, desc=f"Train E{epoch}", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # manual linear warmup
        if global_step < warmup_steps:
            lr_scale = float(global_step + 1) / float(max(1, warmup_steps))
            for g in optimizer.param_groups:
                g["lr"] = base_lr * lr_scale

        if AMP_ON_CUDA and GradScaler is not None and device.type == "cuda":
            with autocast(enabled=True):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # no AMP
            with torch.no_grad() if False else torch.enable_grad():
                logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * xb.size(0)
        running_correct += (preds == yb).sum().item()
        running_total += xb.size(0)
        global_step += 1
        pbar.set_postfix(loss=running_loss / running_total, acc=running_correct / running_total, lr=optimizer.param_groups[0]["lr"])
    return running_loss / running_total, running_correct / running_total, global_step

@torch.no_grad()
def evaluate(model, dl, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dl, desc="Eval", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        if AMP_ON_CUDA and GradScaler is not None and device.type == "cuda":
            with autocast(enabled=True):
                logits = model(xb)
                loss = criterion(logits, yb)
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * xb.size(0)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total

# -------------------------
# Full training pipeline
# -------------------------
def train_and_evaluate(epochs=5, base_lr=3e-4, warmup_steps=500, early_stop_patience=5, compile_model_on_cuda=True):
    model = TinyCNN().to(device)

    # Only attempt torch.compile on CUDA — Inductor + MPS/CPU have issues
    compiled = False
    if compile_model_on_cuda and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            compiled = True
            print("Model compiled with torch.compile (CUDA).")
        except Exception as e:
            print("torch.compile failed — continuing without compile:", e)
            compiled = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    total_steps = epochs * len(train_dl)
    warmup = min(warmup_steps, total_steps)
    scheduler = make_cosine_with_warmup(optimizer, warmup, total_steps, eta_min=1e-5)

    scaler = GradScaler(enabled=(device.type == "cuda")) if AMP_ON_CUDA and device.type == "cuda" else None

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    # tiny overfit sanity check first
    print("Running tiny-subset overfit sanity (1 epoch)...")
    train_one_epoch(model, tiny_dl, optimizer, criterion, scaler, epoch=0, global_step=global_step, warmup_steps=warmup, base_lr=base_lr)

    print("Starting full training...")
    t0 = time.time()
    for epoch in range(epochs):
        train_loss, train_acc, global_step = train_one_epoch(model, train_dl, optimizer, criterion, scaler, epoch+1, global_step, warmup, base_lr)
        # step scheduler AFTER batch updates for cosine decay portion
        # For simplicity, call scheduler.step() once per epoch reduced by warmup steps:
        # we will step scheduler by number of batches processed since last scheduler call.
        # Simpler approach: call scheduler.step() each epoch (coarser) — acceptable for demo.
        if global_step >= warmup:
            # call scheduler.step() once per epoch (coarse)
            scheduler.step()

        val_loss, val_acc = evaluate(model, val_dl, criterion)

        print(f"Epoch {epoch+1}/{epochs} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")

        # checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, "tinycnn_best.pt")
            print("Saved best checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered")
                break
    t1 = time.time()
    print(f"Training done in {(t1-t0):.1f}s (compiled={compiled})")
    return model

# -------------------------
# Run training (reduced epochs for quick test)
# -------------------------
if __name__ == "__main__":
    # reduce epochs for quick run during debugging
    trained_model = train_and_evaluate(epochs=5, base_lr=3e-4, warmup_steps=200, early_stop_patience=3,
                                       compile_model_on_cuda=True)

    # Load best and export ONNX
    ck = torch.load("tinycnn_best.pt", map_location=device)
    model = TinyCNN().to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # Export ONNX (use CPU if MPS/CPU for portability)
    onnx_device = torch.device("cpu") if device.type != "cuda" else device
    example = torch.randn(1, 3, 32, 32).to(onnx_device)
    try:
        torch.onnx.export(model.to(onnx_device), example, "tinycnn.onnx", opset_version=17,
                          input_names=["input"], output_names=["output"])
        print("Exported tinycnn.onnx")
    except Exception as e:
        print("ONNX export failed:", e)
