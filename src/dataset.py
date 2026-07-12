from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset


class FootUlcerDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.transform = transform

        self.image_paths = sorted(list(self.images_dir.glob("*")))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.masks_dir:
            mask_path = self.masks_dir / image_path.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            mask = (mask > 127).astype("float32")
        else:
            mask = None

        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return image, mask, image_path.name

        return image, image_path.name