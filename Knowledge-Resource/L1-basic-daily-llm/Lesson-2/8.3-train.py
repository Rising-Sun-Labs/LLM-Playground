import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import ImageCSVDataset

if __name__=="__main__":
    transform = T.Compose(
        [
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )

    dataset = ImageCSVDataset(
        csv_file ="data/labels.csv",
        img_dir = "data/images",
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle = True,
        num_workers = 0,    # safer on macOS
        pin_memory=False
    )

    xb, yb = next(iter(dataloader))
    print("Image batch shape:", xb.shape)
    print("Label batch shape:", yb.shape)
    print("Labels:", yb)