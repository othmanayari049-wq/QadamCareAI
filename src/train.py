from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from tqdm import tqdm

from dataset import FootUlcerDataset
from model import build_model
from metrics import dice_score, iou_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "raw" / "Foot Ulcer Segmentation Challenge"

TRAIN_IMAGES = DATASET / "train" / "images"
TRAIN_LABELS = DATASET / "train" / "labels"
VAL_IMAGES = DATASET / "validation" / "images"
VAL_LABELS = DATASET / "validation" / "labels"

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_transforms(image_size=256):
    train_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ])

    val_tf = A.Compose([
        A.Resize(image_size, image_size),
    ])

    return train_tf, val_tf


def train_one_model(model_name, architecture, encoder, epochs=5, batch_size=8, image_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining {model_name} on {device}")

    train_tf, val_tf = get_transforms(image_size)

    train_ds = FootUlcerDataset(TRAIN_IMAGES, TRAIN_LABELS, transform=train_tf)
    val_ds = FootUlcerDataset(VAL_IMAGES, VAL_LABELS, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(architecture, encoder).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_dice = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} train"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_dice = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} val"):
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                probs = torch.sigmoid(logits)

                val_dice += dice_score(probs, masks).item()
                val_iou += iou_score(probs, masks).item()

        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"Epoch {epoch}: loss={train_loss:.4f}, dice={val_dice:.4f}, iou={val_iou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            save_path = MODEL_DIR / f"{model_name}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model: {save_path}")

    total_time = time.time() - start_time
    print(f"\nFinished {model_name}. Best Dice: {best_dice:.4f}. Time: {total_time/60:.1f} min")


if __name__ == "__main__":
    train_one_model(
        model_name="unet_efficientnet_b0_25epochs",
        architecture="unet",
        encoder="efficientnet-b0",
        epochs=25,
        batch_size=8,
        image_size=256,
    )