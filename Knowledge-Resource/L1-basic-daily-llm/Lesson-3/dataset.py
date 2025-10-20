# Custom Dataset

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageCSVDataset(Dataset):
    """
    A custom dataset that:
    - Reads image file paths from a directory
    - Reads labels from a CSV file
    - Applies optional transforms to images
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        label = int(row['label'])
        if self.transform:
            image = self.transform(image)
        return image, label

    # Loads image and label based on index
    # Applies Resize and ToTensor transforms
    # Returns (image_tensor, label)
