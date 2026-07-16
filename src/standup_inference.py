from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights


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
    def __init__(self):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.rgb_model = models.efficientnet_b0(weights=weights)
        self.thermal_model = models.efficientnet_b0(weights=weights)

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
    def __init__(self, num_classes=3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.rgb_model = models.efficientnet_b0(weights=weights)
        self.thermal_model = models.efficientnet_b0(weights=weights)

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


_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_binary_model = None
_binary_device = None
_risk_model = None
_risk_device = None


def _load_image(image_path):
    return _transform(Image.open(image_path).convert("RGB")).unsqueeze(0)


def _load_state_dict(checkpoint_path, map_location):
    data = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(data, dict) and "state_dict" in data:
        return data["state_dict"]
    if isinstance(data, dict) and "model_state_dict" in data:
        return data["model_state_dict"]
    return data


def load_dm_control_model(checkpoint_path=DM_CONTROL_CHECKPOINT):
    global _binary_model, _binary_device
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None, None

    if _binary_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionEfficientNetB0Binary()
        state = _load_state_dict(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        _binary_model = model
        _binary_device = device

    return _binary_model, _binary_device


def load_risk_model(checkpoint_path=RISK_CHECKPOINT):
    global _risk_model, _risk_device
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None, None

    if _risk_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionEfficientNetB0Risk(num_classes=3)
        state = _load_state_dict(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        _risk_model = model
        _risk_device = device

    return _risk_model, _risk_device


def analyze_dm_control_pattern(rgb_path, thermal_path, checkpoint_path=DM_CONTROL_CHECKPOINT):
    model, device = load_dm_control_model(checkpoint_path)
    if model is None:
        return {
            "model_available": False,
            "predicted_pattern": "Model checkpoint not available",
            "diabetic_foot_pattern_probability": None,
            "healthy_control_pattern_probability": None,
            "checkpoint_expected_path": str(Path(checkpoint_path)),
            "summary": (
                "The STANDUP RGB+thermal fusion checkpoint is not available locally. "
                "Download best_model.pth from Kaggle and save it as "
                f"{Path(checkpoint_path)}."
            ),
            "safety_note": SAFETY_NOTE,
        }

    rgb = _load_image(rgb_path).to(device)
    thermal = _load_image(thermal_path).to(device)

    with torch.no_grad():
        logit = model(rgb, thermal)
        dm_prob = float(torch.sigmoid(logit)[0].item())

    control_prob = 1.0 - dm_prob
    predicted = (
        "Dataset-defined diabetic-foot-like pattern"
        if dm_prob >= 0.5
        else "Dataset-defined healthy/control-like pattern"
    )

    return {
        "model_available": True,
        "model_name": "STANDUP RGB+thermal EfficientNet-B0 fusion classifier",
        "predicted_pattern": predicted,
        "diabetic_foot_pattern_probability": round(dm_prob, 4),
        "healthy_control_pattern_probability": round(control_prob, 4),
        "threshold": 0.5,
        "summary": (
            f"The paired RGB/thermal image pattern is classified as: {predicted}. "
            f"Dataset-defined diabetic-foot-like probability: {dm_prob:.1%}."
        ),
        "safety_note": SAFETY_NOTE,
    }


def analyze_r0_r1_r2_risk_pattern(rgb_path, thermal_path, checkpoint_path=RISK_CHECKPOINT):
    model, device = load_risk_model(checkpoint_path)
    labels = ["R0 low-risk pattern", "R1 medium-risk pattern", "R2 high-risk pattern"]

    if model is None:
        return {
            "model_available": False,
            "predicted_risk_pattern": "Risk model checkpoint not available",
            "probabilities": None,
            "checkpoint_expected_path": str(Path(checkpoint_path)),
            "summary": (
                "The R0/R1/R2 STANDUP risk-pattern checkpoint is not available locally yet. "
                "Train the diabetic-subset fusion model on Kaggle, then save best_model.pth as "
                f"{Path(checkpoint_path)}."
            ),
            "safety_note": SAFETY_NOTE,
        }

    rgb = _load_image(rgb_path).to(device)
    thermal = _load_image(thermal_path).to(device)

    with torch.no_grad():
        logits = model(rgb, thermal)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().tolist()

    pred_idx = int(max(range(len(probs)), key=lambda idx: probs[idx]))
    probability_dict = {labels[i]: round(float(probs[i]), 4) for i in range(3)}

    return {
        "model_available": True,
        "model_name": "STANDUP R0/R1/R2 RGB+thermal EfficientNet-B0 fusion classifier",
        "predicted_risk_pattern": labels[pred_idx],
        "probabilities": probability_dict,
        "summary": (
            f"The diabetic-foot risk-pattern classifier selected {labels[pred_idx]} "
            f"with probability {probs[pred_idx]:.1%}."
        ),
        "safety_note": SAFETY_NOTE,
    }
