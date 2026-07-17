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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "app