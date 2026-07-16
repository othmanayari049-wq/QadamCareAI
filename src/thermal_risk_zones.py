from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


SAFETY_NOTE = (
    "Thermal risk-zone highlighting is a screening-support visualization only. "
    "It highlights high-monitoring zones based on relative image intensity, not true temperature measurement, "
    "future ulcer prediction, diagnosis, or treatment guidance."
)


def _read_gray(image_path):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read thermal image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return rgb, gray


def _normalize(gray):
    min_value = float(np.min(gray))
    max_value = float(np.max(gray))
    if max_value - min_value < 1e-6:
        return np.zeros_like(gray, dtype=np.float32)
    return (gray - min_value) / (max_value - min_value)


def _quality_flags(gray):
    h, w = gray.shape[:2]
    blur_score = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
    contrast = float(np.std(gray))

    warnings = []
    if min(h, w) < 224:
        warnings.append("Thermal image resolution is low; retake or use caution.")
    if contrast < 8:
        warnings.append("Thermal contrast is low; hotspot interpretation may be weak.")
    if blur_score < 5:
        warnings.append("Thermal image may be blurry; retake may be needed.")

    return {
        "status": "PASS" if not warnings else "REVIEW_NEEDED",
        "width_px": int(w),
        "height_px": int(h),
        "blur_score": round(blur_score, 3),
        "contrast": round(contrast, 3),
        "warnings": warnings,
    }


def _largest_components(binary_mask, max_regions=5):
    mask = binary_mask.astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []

    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]
        regions.append({
            "region_id": len(regions) + 1,
            "area_pixels": area,
            "bbox_xywh": [x, y, w, h],
            "centroid_xy": [round(float(cx), 2), round(float(cy), 2)],
            "interpretation": "High-monitoring thermal zone for clinician review",
        })

    regions = sorted(regions, key=lambda item: item["area_pixels"], reverse=True)
    return regions[:max_regions]


def create_risk_zone_overlay(rgb, normalized, high_mask, medium_mask, alpha=0.42):
    overlay = rgb.copy()

    # high = red, medium = orange/yellow, low/normal remains original
    overlay[medium_mask] = [255, 180, 0]
    overlay[high_mask] = [255, 0, 0]

    blended = cv2.addWeighted(rgb, 1 - alpha, overlay, alpha, 0)
    return blended


def analyze_thermal_risk_zones(image_path):
    """
    Create high/medium/low monitoring zones from a thermal image using relative intensity.

    This does not predict exact future ulcer locations. It provides a conservative
    monitoring-zone visualization that can be compared with later visits.
    """
    rgb, gray = _read_gray(image_path)
    normalized = _normalize(gray)

    # Percentile-based thresholds are image-relative because public thermograms are often stored as images,
    # not calibrated temperature arrays.
    high_threshold = float(np.percentile(normalized, 90))
    medium_threshold = float(np.percentile(normalized, 75))

    high_mask = normalized >= high_threshold
    medium_mask = (normalized >= medium_threshold) & (normalized < high_threshold)
    low_mask = normalized < medium_threshold

    total_pixels = int(normalized.size)
    high_pixels = int(high_mask.sum())
    medium_pixels = int(medium_mask.sum())

    hot_region_ratio = high_pixels / total_pixels if total_pixels else 0.0
    medium_region_ratio = medium_pixels / total_pixels if total_pixels else 0.0

    h, w = normalized.shape[:2]
    left_mean = float(normalized[:, : w // 2].mean()) if w > 1 else 0.0
    right_mean = float(normalized[:, w // 2 :].mean()) if w > 1 else 0.0
    asymmetry_score = abs(left_mean - right_mean)

    if hot_region_ratio >= 0.08 or asymmetry_score >= 0.10:
        overall_zone = "HIGH_MONITORING"
    elif hot_region_ratio >= 0.03 or asymmetry_score >= 0.05:
        overall_zone = "MEDIUM_MONITORING"
    else:
        overall_zone = "LOW_MONITORING"

    overlay = create_risk_zone_overlay(rgb, normalized, high_mask, medium_mask)
    regions = _largest_components(high_mask, max_regions=5)
    quality = _quality_flags(gray)

    return {
        "thermal_zone_available": True,
        "quality": quality,
        "overall_monitoring_zone": overall_zone,
        "hot_region_ratio": round(float(hot_region_ratio), 4),
        "medium_region_ratio": round(float(medium_region_ratio), 4),
        "asymmetry_score": round(float(asymmetry_score), 4),
        "high_threshold_percentile": 90,
        "medium_threshold_percentile": 75,
        "high_monitoring_regions": regions,
        "original_rgb": rgb,
        "normalized_map": normalized,
        "high_mask": high_mask.astype(np.uint8),
        "medium_mask": medium_mask.astype(np.uint8),
        "low_mask": low_mask.astype(np.uint8),
        "risk_zone_overlay": overlay,
        "summary": (
            f"Thermal analysis found {overall_zone.replace('_', ' ').lower()} based on "
            f"relative hotspot ratio {hot_region_ratio:.1%} and asymmetry score {asymmetry_score:.3f}."
        ),
        "safety_note": SAFETY_NOTE,
    }


def save_thermal_risk_outputs(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    overlay_path = output_dir / f"thermal_risk_zone_overlay_{timestamp}.png"
    high_mask_path = output_dir / f"thermal_high_monitoring_mask_{timestamp}.png"
    medium_mask_path = output_dir / f"thermal_medium_monitoring_mask_{timestamp}.png"
    normalized_path = output_dir / f"thermal_normalized_map_{timestamp}.png"

    cv2.imwrite(str(overlay_path), cv2.cvtColor(result["risk_zone_overlay"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(high_mask_path), result["high_mask"] * 255)
    cv2.imwrite(str(medium_mask_path), result["medium_mask"] * 255)
    cv2.imwrite(str(normalized_path), (result["normalized_map"] * 255).astype(np.uint8))

    return {
        "overlay_path": str(overlay_path),
        "high_mask_path": str(high_mask_path),
        "medium_mask_path": str(medium_mask_path),
        "normalized_path": str(normalized_path),
    }
