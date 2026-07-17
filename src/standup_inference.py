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
    "ST