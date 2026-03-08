"""Dataset wrappers backed by split CSV manifests."""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PlantVillageSplitDataset(Dataset):
    """Loads image path + integer label rows from generated split CSVs."""

    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        """Number of samples in split."""
        return len(self.df)

    def __getitem__(self, idx):
        """Return transformed RGB image tensor and class index."""
        row = self.df.iloc[idx]
        img = Image.open(Path(row["image_path"])).convert("RGB")
        label = int(row["label_idx"])
        if self.transform:
            img = self.transform(img)
        return img, label
