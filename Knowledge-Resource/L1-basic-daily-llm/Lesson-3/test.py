import torch
from PIL import Image
import torchvision.transforms as T
from model import SimpleCNN

# Load Model
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load("cnn_weights.pt", map_location="cpu"))
model.eval()

# Image transform (must match training)
transform = T.Compose(
    [
        T.Resize((128, 128)),
        T.ToTensor()
    ]
)

# Choose image to test
image_path = "data/images/dog1.jpg"
image = Image.open(image_path).convert("RGB")
x = transform(image).unsqueeze(0)   # add batch dimension

# Prediction
with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(1).item()

label_map = {0: "cat", 1: "dog"}
print(f"Predicted class: {label_map[pred]}")

# Loads trained model
# Read a single image
# Applies the same transform
# Predicts and prints label
