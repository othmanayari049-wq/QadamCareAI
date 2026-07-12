from pathlib import Path
import json
import cv2
import torch

from src.thermal_gradcam import (
    load_thermal_gradcam,
    prepare_thermal_image,
    create_gradcam_overlay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "thermal"
    / "threshold_selection"
    / "selected_thermal_threshold.json"
)


def load_thermal_threshold():
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return float(data["selected_threshold"])


class ThermalInferenceEngine:
    def __init__(self):
        self.model = None
        self.gradcam = None
        self.checkpoint = None
        self.device = None
        self.threshold = load_thermal_threshold()

    def load(self):
        if self.model is None:
            (
                self.model,
                self.gradcam,
                self.checkpoint,
                self.device,
            ) = load_thermal_gradcam()

    def predict(self, image_path):
        self.load()

        image_path = Path(image_path)

        image_rgb, image_tensor = prepare_thermal_image(
            image_path,
            image_size=self.checkpoint["image_size"],
        )

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            dm_probability = float(
                torch.softmax(output, dim=1)[0, 1].item()
            )

        predicted_label = (
            "DM Group thermal pattern"
            if dm_probability >= self.threshold
            else "Control Group thermal pattern"
        )

        # Grad-CAM requires gradients, so it runs separately.
        gradcam_result = self.gradcam.generate(
            image_tensor,
            class_index=1,
        )

        heatmap_color, overlay = create_gradcam_overlay(
            image_rgb,
            gradcam_result["heatmap"],
        )

        return {
            "thermal_available": True,
            "dm_probability": round(dm_probability, 4),
            "threshold": self.threshold,
            "predicted_pattern": predicted_label,
            "model_name": "EfficientNet-B0 thermal classifier",
            "attention_map": heatmap_color,
            "attention_overlay": overlay,
            "original_image": image_rgb,
            "summary": (
                f"Thermal classifier probability for the dataset-defined DM Group "
                f"pattern: {dm_probability:.1%}. "
                f"Decision threshold: {self.threshold:.2f}."
            ),
            "safety_note": (
                "This model distinguishes dataset-defined DM Group versus Control "
                "Group plantar thermogram patterns. It does not diagnose diabetes, "
                "diabetic-foot ulcer, infection, wound severity, or actual temperature."
            ),
        }


_thermal_engine = None


def analyze_thermal_with_model(image_path):
    global _thermal_engine

    if _thermal_engine is None:
        _thermal_engine = ThermalInferenceEngine()

    return _thermal_engine.predict(image_path)


def save_attention_images(result, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    original_path = output_folder / "thermal_original.png"
    map_path = output_folder / "thermal_model_attention_map.png"
    overlay_path = output_folder / "thermal_model_attention_overlay.png"

    cv2.imwrite(
        str(original_path),
        cv2.cvtColor(result["original_image"], cv2.COLOR_RGB2BGR),
    )

    cv2.imwrite(
        str(map_path),
        cv2.cvtColor(result["attention_map"], cv2.COLOR_RGB2BGR),
    )

    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(result["attention_overlay"], cv2.COLOR_RGB2BGR),
    )

    return {
        "original_path": str(original_path),
        "attention_map_path": str(map_path),
        "attention_overlay_path": str(overlay_path),
    }
