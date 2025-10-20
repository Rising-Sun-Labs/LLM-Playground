import os
from PIL import Image, ImageDraw
import pandas as pd

# Create folders
os.makedirs("data/images", exist_ok=True)

# Image names and labels
images = [
    ("dog1.jpg", 0),
    ("dog2.jpg", 0),
    ("cat1.jpg", 1),
    ("cat2.jpg", 1),
]

# Generate simple images
for filename, label in images:
    img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 50), filename.split(".")[0],  fill=(0,0,0))
    img.save(os.path.join("data/images", filename))


# Create labels.csv
df = pd.DataFrame(images, columns=["image_path", "label"])
df.to_csv("data/labels.csv", index=False)

print("✅ Sample images and CSV generated in ./data/")
