from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

DM_CONTROL_CHECKPOINT = MODEL_DIR / "standup_rgb_thermal_fusion_efficientnetb0.pth"
RISK_CHECKPOINT = MODEL_DIR / "standup_r0_r1_r2_fusion_efficientnetb0.pth"

SAFETY_NOTE = (
    "STANDUP image-pattern models classify dataset-defined RGB/thermal foot-image patterns. "
    "They do not diagnose diabetes, estimate blood glucose, predict exact future ulcer location, "
    "or replace clinician assessment."
)


class FusionEfficientNetB0Binary(nn.Module):
    """Architecture used by the locally trained DM/control fusion checkpoint."""

    def __init__(self):
        super().__init__()
        self.rgb_model = models.efficientnet_b0(weights=None)
        self.thermal_model = models.efficientnet_b0(weights=None)

        rgb_features = self.rgb_model.classifier[1].in_features
        thermal_features = self.thermal_model.classifier[1].in_features
        self.rgb_model.classifier = nn.Identity()
        self.thermal_model.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(rgb_features + thermal_features, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, rgb, thermal):
        rgb_features = self.rgb_model(rgb)
        thermal_features = self.thermal_model(thermal)
        fused = torch.cat([rgb_features, thermal_features], dim=1)
        return self.classifier(fused).squeeze(1)


class FusionEfficientNetB0Risk(nn.Module):
    """Original simple R0/R1/R2 architecture.

    Newer experimental checkpoints may use another classifier architecture. Those
    checkpoints are rejected safely instead of crashing the Streamlit application.
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.rgb_model = models.efficientnet_b0(weights=None)
        self.thermal_model = models.efficientnet_b0(weights=None)

        rgb_features = self.rgb_model.classifier[1].in_features
        thermal_features = self.thermal_model.classifier[1].in_features
        self.rgb_model.classifier = nn.Identity()
        self.thermal_model.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(rgb_features + thermal_features, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, rgb, thermal):
        rgb_features = self.rgb_model(rgb)
        thermal_features = self.thermal_model(thermal)
        fused = torch.cat([rgb_features, thermal_features], dim=1)
        return self.classifier(fused)


_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

_binary_model = None
_binary_device = None
_binary_load_error = None

_risk_model = None
_risk_device = None
_risk_load_error = None


def _load_image(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    return _IMAGE_TRANSFORM(image).unsqueeze(0)


def _load_state_dict(checkpoint_path, map_location):
    data = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(data, dict) and "state_dict" in data:
        state = data["state_dict"]
    elif isinstance(data, dict) and "model_state_dict" in data:
        state = data["model_state_dict"]
    else:
        state = data

    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a valid PyTorch state_dict.")

    # Support checkpoints saved from DataParallel.
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}

    return state


def load_dm_control_model(checkpoint_path=DM_CONTROL_CHECKPOINT):
    global _binary_model, _binary_device, _binary_load_error

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        _binary_load_error = f"Checkpoint not found: {checkpoint_path}"
        return None, None

    if _binary_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionEfficientNetB0Binary()

        try:
            state = _load_state_dict(checkpoint_path, map_location=device)
            model.load_state_dict(state, strict=True)
        except (RuntimeError, ValueError, KeyError, OSError) as error:
            _binary_load_error = (
                "The DM/control checkpoint could not be loaded because its architecture or file "
                f"format does not match the application model. Details: {error}"
            )
            return None, None

        model.to(device)
        model.eval()
        _binary_model = model
        _binary_device = device
        _binary_load_error = None

    return _binary_model, _binary_device


def load_risk_model(checkpoint_path=RISK_CHECKPOINT):
    global _risk_model, _risk_device, _risk_load_error

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        _risk_load_error = f"Checkpoint not found: {checkpoint_path}"
        return None, None

    if _risk_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionEfficientNetB0Risk(num_classes=3)

        try:
            state = _load_state_dict(checkpoint_path, map_location=device)
            model.load_state_dict(state, strict=True)
        except (RuntimeError, ValueError, KeyError, OSError) as error:
            _risk_load_error = (
                "The experimental R0/R1/R2 checkpoint uses an incompatible architecture and was "
                "skipped safely. The remaining multimodal analysis can continue. "
                f"Details: {error}"
            )
            return None, None

        model.to(device)
        model.eval()
        _risk_model = model
        _risk_device = device
        _risk_load_error = None

    return _risk_model, _risk_device


def analyze_dm_control_pattern(
    rgb_path,
    thermal_path,
    checkpoint_path=DM_CONTROL_CHECKPOINT,
):
    model, device = load_dm_control_model(checkpoint_path)

    if model is None:
        return {
            "model_available": False,
            "predicted_pattern": "DM/control model unavailable",
            "diabetic_foot_pattern_probability": None,
            "healthy_control_pattern_probability": None,
            "checkpoint_expected_path": str(Path(checkpoint_path)),
            "summary": _binary_load_error
            or "The STANDUP RGB+thermal DM/control checkpoint is unavailable.",
            "safety_note": SAFETY_NOTE,
        }

    rgb = _load_image(rgb_path).to(device)
    thermal = _load_image(thermal_path).to(device)

    with torch.inference_mode():
        logit = model(rgb, thermal)
        dm_probability = float(torch.sigmoid(logit)[0].item())

    control_probability = 1.0 - dm_probability
    predicted_pattern = (
        "Dataset-defined diabetic-foot-like pattern"
        if dm_probability >= 0.5
        else "Dataset-defined healthy/control-like pattern"
    )

    return {
        "model_available": True,
        "model_name": "STANDUP RGB+thermal EfficientNet-B0 fusion classifier",
        "predicted_pattern": predicted_pattern,
        "diabetic_foot_pattern_probability": round(dm_probability, 4),
        "healthy_control_pattern_probability": round(control_probability, 4),
        "threshold": 0.5,
        "summary": (
            f"The paired RGB/thermal image pattern is classified as {predicted_pattern}. "
            f"Dataset-defined diabetic-foot-like probability: {dm_probability:.1%}."
        ),
        "safety_note": SAFETY_NOTE,
    }


def analyze_r0_r1_r2_risk_pattern(
    rgb_path,
    thermal_path,
    checkpoint_path=RISK_CHECKPOINT,
):
    labels = [
        "R0 low-risk pattern",
        "R1 medium-risk pattern",
        "R2 high-risk pattern",
    ]
    model, device = load_risk_model(checkpoint_path)

    if model is None:
        return {
            "model_available": False,
            "predicted_risk_pattern": "Experimental model unavailable",
            "probabilities": None,
            "checkpoint_expected_path": str(Path(checkpoint_path)),
            "summary": _risk_load_error
            or "The experimental R0/R1/R2 checkpoint is unavailable.",
            "safety_note": (
                SAFETY_NOTE
                + " R0/R1/R2 performance was not strong enough for reliable use and remains experimental."
            ),
        }

    rgb = _load_image(rgb_path).to(device)
    thermal = _load_image(thermal_path).to(device)

    with torch.inference_mode():
        logits = model(rgb, thermal)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    predicted_index = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    probability_dict = {
        labels[index]: round(float(probabilities[index]), 4)
        for index in range(len(labels))
    }

    return {
        "model_available": True,
        "model_name": "Experimental STANDUP R0/R1/R2 RGB+thermal classifier",
        "predicted_risk_pattern": labels[predicted_index],
        "probabilities": probability_dict,
        "summary": (
            f"The experimental classifier selected {labels[predicted_index]} "
            f"with probability {probabilities[predicted_index]:.1%}."
        ),
        "safety_note": (
            SAFETY_NOTE
            + " R0/R1/R2 performance was not strong enough for reliable use and remains experimental."
        ),
    }
