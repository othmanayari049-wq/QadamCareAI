from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import albumentations as A

from model import build_model
from quality import assess_image_quality
from features import extract_lesion_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "unet_efficientnet_b0_25epochs_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "app_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_segmentation_model = None
_segmentation_device = None

SAFETY_NOTE = (
    "The RGB ulcer segmentation model highlights visible wound/ulcer-like regions for "
    "screening-support and documentation only. It does not diagnose diabetic-foot ulcer, "
    "infection, ischemia, depth, Wagner grade, or treatment need."
)


def load_segmentation_model(model_path=MODEL_PATH):
    global _segmentation_model, _segmentation_device
    model_path = Path(model_path)

    if not model_path.exists():
        return None, None

    if _segmentation_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model("unet", "efficientnet-b0")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        _segmentation_model = model
        _segmentation_device = device

    return _segmentation_model, _segmentation_device


def analyze_ulcer_segmentation(image_path, model_path=MODEL_PATH, image_size=256):
    image_path = Path(image_path)
    model, device = load_segmentation_model(model_path)

    if model is None:
        return {
            "model_available": False,
            "visible_ulcer_like_region_detected": None,
            "number_of_regions": None,
            "total_area_pixels": None,
            "confidence": None,
            "checkpoint_expected_path": str(Path(model_path)),
            "summary": (
                "The FUSeg ulcer segmentation checkpoint is not available locally. "
                f"Save the trained checkpoint as {Path(model_path)}."
            ),
            "safety_note": SAFETY_NOTE,
        }

    quality = assess_image_quality(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Unable to read RGB image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = A.Resize(image_size, image_size)(image=image_rgb)["image"]

    x = (
        torch.tensor(resized, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    ).to(device)

    start = datetime.now()
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    inference_time = (datetime.now() - start).total_seconds()

    mask = (prob > 0.5).astype(np.uint8)
    features = extract_lesion_features(mask, prob_map=prob)
    confidence = float(prob[mask == 1].mean()) if mask.sum() > 0 else 0.0

    overlay = resized.copy()
    overlay[mask == 1] = [255, 0, 0]
    overlay = cv2.addWeighted(resized, 0.65, overlay, 0.35, 0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_path = OUTPUT_DIR / f"ulcer_original_{timestamp}.png"
    mask_path = OUTPUT_DIR / f"ulcer_mask_{timestamp}.png"
    overlay_path = OUTPUT_DIR / f"ulcer_overlay_{timestamp}.png"

    cv2.imwrite(str(original_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask * 255)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    detected = features["number_of_lesions"] > 0
    summary = (
        f"Detected {features['number_of_lesions']} visible wound/ulcer-like region(s), "
        f"with total predicted area {features['total_area_pixels']} pixels."
        if detected
        else "No clear visible wound/ulcer-like region was detected by the segmentation model."
    )

    return {
        "model_available": True,
        "visible_ulcer_like_region_detected": detected,
        "number_of_regions": int(features["number_of_lesions"]),
        "total_area_pixels": int(features["total_area_pixels"]),
        "confidence": round(float(confidence), 4),
        "quality": quality,
        "features": features,
        "image": resized,
        "mask": mask,
        "overlay": overlay,
        "original_path": str(original_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "inference_time_seconds": round(float(inference_time), 4),
        "summary": summary,
        "safety_note": SAFETY_NOTE,
    }
