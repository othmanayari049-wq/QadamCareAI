from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

from model import build_model
from quality import assess_image_quality
from features import extract_lesion_features
from report import generate_clinician_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "raw" / "Foot Ulcer Segmentation Challenge"

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "unet_efficientnet_b0_25epochs_best.pth"

SAVE_DIR = PROJECT_ROOT / "outputs" / "predictions"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def predict_one(image_path, mask_path=None, image_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quality_result = assess_image_quality(image_path)

    model = build_model("unet", "efficientnet-b0")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = A.Compose([A.Resize(image_size, image_size)])
    resized = transform(image=image_rgb)["image"]

    x = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    pred_mask = (prob > 0.5).astype(np.uint8)

    feature_result = extract_lesion_features(pred_mask, prob_map=prob)

    wound_area_pixels = int(pred_mask.sum())
    confidence = float(prob[pred_mask == 1].mean()) if wound_area_pixels > 0 else 0.0

    true_mask = None
    if mask_path is not None:
        true_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if true_mask is not None:
            true_mask = cv2.resize(true_mask, (image_size, image_size))
            true_mask = (true_mask > 127).astype(np.uint8)

    overlay = resized.copy()
    overlay[pred_mask == 1] = [255, 0, 0]
    blended = cv2.addWeighted(resized, 0.65, overlay, 0.35, 0)

    save_path = SAVE_DIR / f"prediction_{Path(image_path).stem}.png"

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(resized)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(true_mask if true_mask is not None else np.zeros_like(pred_mask), cmap="gray")
    plt.title("Expert mask")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("AI prediction")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(blended)
    plt.title("AI overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    report_path = generate_clinician_report(
        image_name=Path(image_path).name,
        quality_result=quality_result,
        feature_result=feature_result,
        confidence=confidence,
        output_dir=SAVE_DIR,
    )

    print("Image:", Path(image_path).name)
    print("Quality status:", quality_result["status"])
    print("Predicted wound area pixels:", wound_area_pixels)
    print("Average model confidence:", round(confidence, 4))
    print("Number of lesions:", feature_result["number_of_lesions"])

    for lesion in feature_result["lesions"]:
        print(
            f"Lesion {lesion['lesion_id']}: "
            f"area={lesion['area_pixels']} px, "
            f"confidence={lesion['mean_confidence']}"
        )

    print("Saved image:", save_path)
    print("Saved report:", report_path)


if __name__ == "__main__":
    for name in ["0001.png", "0002.png", "0003.png", "0004.png", "0005.png"]:
        image_path = DATASET / "validation" / "images" / name
        mask_path = DATASET / "validation" / "labels" / name
        predict_one(image_path, mask_path)
        print("-" * 60)