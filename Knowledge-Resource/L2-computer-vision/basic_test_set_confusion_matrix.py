import torch
import matplotlib.pyplot as plt
import numpy as np


# Load best
state = torch.load("tinycnn_best.pt", map_locatin=device)
model.load_state_dict(state["model"]); model.eval()

all_pred, all_true =[], []
with torch.no_grad():
    for xb, yb in test_dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        all_pred.append(pred.cpu()); all_true.append(yb.cpu())

    
y_pred = torch.cat(all_pred).numpy()
y_true = torch.cat(all_true).numpy()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
print("test acc:", (y_pred==y_true).mean())

# Qucik plot
fig = plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest')
plt.title("CIFAR-10 Confusion Matrix"); plt.xlabel("pred"); plt.ylabel("true")
plt.tight_layout(); plt.show()
