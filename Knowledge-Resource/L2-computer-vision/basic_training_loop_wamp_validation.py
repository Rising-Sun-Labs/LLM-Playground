import torch.optim as optim
from tqdm import tqdm

crit = nn.CrossEntropyLoss()
opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

def run_epoch(dl, train=True):
    model.train(train)
    total, correct, total_loss = 0, 0, 0.0
    for xb, yb in tqdm(dl, leave=False):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        if train: opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            logits = model(xb)
            loss = crit(logits, yb)
        if train:
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        total_loss += loss.item()*xb.size(0)
        pred = logits.argmax(1)
        correct += (pred==yb).sum().item()
        total+=xb.size(0)
    return total_loss/total, correct/total


best_val = 0.0
for epoch in range(10):
    tr_loss, tr_acc = run_epoch(train_dl, train=True)
    val_loss, val_acc = run_epoch(val_dl, train=False)
    print(f"epoch {epoch:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {val_loss:.3f} acc {val_acc:.3f}")
    if val_acc > best_val:
        best_val=val_acc
        torch.save({"model": model.state_dict()}, "tinycnn_best.pt")
        print(" Saved checkpoint (best so far)")



# checkpoints
# you see training/validation accuracy imporving
# a tinycnn_best.pt checkpoint is saved when validation improves
