from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
import albumentations as A
from tqdm import tqdm

from dataset import FootUlcerDataset
from model import build_model
from metrics import dice_score, iou_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "raw" / "Foot Ulcer Segmentation Challenge"

VAL_IMAGES = DATASET / "validation" / "images"
VAL_LABELS = DATASET / "validation" / "labels"

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "unet_efficientnet_b0_25epochs_best.pth"


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = A.Compose([A.Resize(256, 256)])

    val_ds = FootUlcerDataset(VAL_IMAGES, VAL_LABELS, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model("unet", "efficientnet-b0")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    dice_total = 0
    iou_total = 0
    precision_total = 0
    recall_total = 0
    total_time = 0

    eps = 1e-7

    with torch.no_grad():
        for images, masks, names in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            start = time.time()
            logits = model(images)
            probs = torch.sigmoid(logits)
            total_time += time.time() - start

            preds = (probs > 0.5).float()

            dice_total += dice_score(probs, masks).item()
            iou_total += iou_score(probs, masks).item()

            tp = (preds * masks).sum()
            fp = (preds * (1 - masks)).sum()
            fn = ((1 - preds) * masks).sum()

            precision = (tp + eps) / (tp + fp + eps)
            recall = (tp + eps) / (tp + fn + eps)

            precision_total += precision.item()
            recall_total += recall.item()

    n = len(val_loader)

    print("\nFinal validation results")
    print("=" * 40)
    print(f"Images evaluated      : {n}")
    print(f"Dice score            : {dice_total / n:.4f}")
    print(f"IoU score             : {iou_total / n:.4f}")
    print(f"Precision             : {precision_total / n:.4f}")
    print(f"Recall                : {recall_total / n:.4f}")
    print(f"Avg inference time/img: {total_time / n:.4f} seconds")


if __name__ == "__main__":
    evaluate()