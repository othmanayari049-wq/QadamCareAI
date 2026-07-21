from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


SAFETY_NOTE = (
    "This module describes relative grayscale image intensity inside user-supplied foot regions. "
    "It does not measure calibrated temperature, diagnose disease, determine clinical risk, or "
    "predict a future ulcer location."
)


def _read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image.astype(np.float32)


def _read_mask(path, shape):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to read ROI mask: {path}")
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask >= 127


def _normalize_within_mask(gray, mask):
    values = gray[mask]
    if values.size == 0:
        raise ValueError("The supplied foot ROI mask contains no foreground pixels.")
    lo = float(np.percentile(values, 1))
    hi = float(np.percentile(values, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(gray, dtype=np.float32)
    normalized = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return normalized


def _crop_to_mask(image, mask):
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("The supplied foot ROI mask contains no foreground pixels.")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _masked_zone_mean(image, mask, y0, y1):
    zone_image = image[y0:y1]
    zone_mask = mask[y0:y1]
    values = zone_image[zone_mask]
    return float(values.mean()) if values.size else None


def _zone_asymmetry(normalized, left_mask, right_mask):
    left_img, left_roi = _crop_to_mask(normalized, left_mask)
    right_img, right_roi = _crop_to_mask(normalized, right_mask)

    target_h = max(left_img.shape[0], right_img.shape[0], 3)
    target_w = max(left_img.shape[1], right_img.shape[1], 3)

    left_img = cv2.resize(left_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    left_roi = cv2.resize(left_roi.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    right_img = cv2.resize(right_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    right_roi = cv2.resize(right_roi.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Mirror the right plantar region for approximate corresponding-zone comparison.
    right_img = np.fliplr(right_img)
    right_roi = np.fliplr(right_roi)

    boundaries = [0, target_h // 3, (2 * target_h) // 3, target_h]
    names = ["toes/forefoot", "midfoot", "heel"]
    output = {}
    for index, name in enumerate(names):
        y0, y1 = boundaries[index], boundaries[index + 1]
        left_mean = _masked_zone_mean(left_img, left_roi, y0, y1)
        right_mean = _masked_zone_mean(right_img, right_roi, y0, y1)
        if left_mean is None or right_mean is None:
            output[name] = None
        else:
            output[name] = round(abs(left_mean - right_mean), 4)
    return output


def analyze_relative_thermal_with_rois(thermal_path, left_mask_path=None, right_mask_path=None):
    gray = _read_gray(thermal_path)
    h, w = gray.shape[:2]

    if left_mask_path is None or right_mask_path is None:
        values = gray.reshape(-1)
        upper_threshold = float(np.percentile(values, 90))
        middle_threshold = float(np.percentile(values, 75))
        upper_ratio = float((gray >= upper_threshold).mean())
        middle_ratio = float(((gray >= middle_threshold) & (gray < upper_threshold)).mean())
        return {
            "roi_status": "NOT_AVAILABLE",
            "analysis_scope": "Complete submitted thermal frame",
            "upper_decile_coverage_percent": round(upper_ratio * 100, 2),
            "middle_intensity_coverage_percent": round(middle_ratio * 100, 2),
            "whole_frame_left_right_difference": round(abs(float(gray[:, : w // 2].mean()) - float(gray[:, w // 2 :].mean())) / 255.0, 4) if w > 1 else 0.0,
            "anatomical_asymmetry_available": False,
            "zone_asymmetry": None,
            "clinical_monitoring_level": "Not determined",
            "temperature_measurement": "Not available from the displayed grayscale image alone",
            "overlay": None,
            "notes": [
                "Validated left-foot and right-foot ROI masks were not supplied.",
                "Whole-frame intensity coverage may include the participant, clothing, and background.",
                "No anatomical foot-zone label or clinical monitoring category was generated.",
            ],
            "safety_note": SAFETY_NOTE,
        }

    left_mask = _read_mask(left_mask_path, gray.shape)
    right_mask = _read_mask(right_mask_path, gray.shape)
    overlap = left_mask & right_mask
    if overlap.any():
        raise ValueError("Left and right foot ROI masks overlap. Correct the masks and retry.")

    combined = left_mask | right_mask
    normalized = _normalize_within_mask(gray, combined)
    roi_values = normalized[combined]
    upper_threshold = float(np.percentile(roi_values, 90))
    middle_threshold = float(np.percentile(roi_values, 75))
    upper_mask = combined & (normalized >= upper_threshold)
    middle_mask = combined & (normalized >= middle_threshold) & (normalized < upper_threshold)

    upper_ratio = float(upper_mask.sum() / combined.sum())
    middle_ratio = float(middle_mask.sum() / combined.sum())
    zones = _zone_asymmetry(normalized, left_mask, right_mask)

    base = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    colour = base.copy()
    colour[middle_mask] = [255, 180, 0]
    colour[upper_mask] = [255, 0, 0]
    overlay = cv2.addWeighted(base, 0.62, colour, 0.38, 0)
    overlay[~combined] = (overlay[~combined] * 0.25).astype(np.uint8)

    return {
        "roi_status": "AVAILABLE",
        "analysis_scope": "User-supplied left and right plantar-foot ROI masks",
        "upper_decile_coverage_percent": round(upper_ratio * 100, 2),
        "middle_intensity_coverage_percent": round(middle_ratio * 100, 2),
        "whole_frame_left_right_difference": None,
        "anatomical_asymmetry_available": True,
        "zone_asymmetry": zones,
        "clinical_monitoring_level": "Not determined",
        "temperature_measurement": "Not available from the displayed grayscale image alone",
        "overlay": overlay,
        "notes": [
            "Upper-decile coverage is defined by the brightest 10% of pixels inside the supplied foot ROIs.",
            "Zone asymmetry is an approximate relative-intensity comparison after resizing and mirroring the right ROI.",
            "These values are descriptive and are not validated clinical thresholds.",
        ],
        "safety_note": SAFETY_NOTE,
    }


def save_relative_thermal_overlay(result, output_dir):
    overlay = result.get("overlay")
    if overlay is None:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"relative_thermal_roi_overlay_{stamp}.png"
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return str(path)
