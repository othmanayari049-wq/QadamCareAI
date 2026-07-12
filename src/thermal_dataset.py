from pathlib import Path
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset


class ThermalFootDataset(Dataset):
    def __init__(self, csv_path, image_size=224, augment=False):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(row["image_path"])
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = cv2.flip(image, 1)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return image, label, row["patient_id"], row["filename"]
