import math
from typing import Dict, Optional

import numpy as np


SAFETY_NOTE = (
    "Measurement-based monitoring support is not a diagnosis, treatment recommendation, "
    "future-ulcer prediction, or replacement for clinician assessment. Scores and zones are "
    "prototype documentation-support outputs based on image-derived measurements."
)


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def _percent(value):
    return round(float(value) * 100.0, 2)


def _zone_from_xy(x, y, width, height):
    # Plantar-foot approximation when no anatomical landmark model is available.
    # y position gives longitudinal region; x position gives side.
    y_ratio = y / max(height, 1)
    x_ratio = x / max(width, 1)

    if y_ratio < 0.20:
        longitudinal = "toe region"
    elif y_ratio < 0.45:
        longitudinal = "forefoot"
    elif y_ratio < 0.70:
        longitudinal = "midfoot"
    else:
        longitudinal = "heel"

    side = "left side" if x_ratio < 0.45 else "right side" if x_ratio > 0.55 else "central region"
    return f"{longitudinal}, {side}"


def _dominant_mask_zone(mask):
    if mask is None:
        return "Not available"
    mask = np.asarray(mask)
    if mask.size == 0 or mask.sum() == 0:
        return "No highlighted zone"
    ys, xs = np.where(mask > 0)
    h, w = mask.shape[:2]
    return _zone_from_xy(float(xs.mean()), float(ys.mean()), w, h)


def compute_image_quality_score(rgb_quality: Dict) -> Dict:
    metrics = rgb_quality.get("metrics", {}) if isinstance(rgb_quality, dict) else {}
    blur = _safe_float(metrics.get("blur_score"), 25.0)
    brightness = _safe_float(metrics.get("brightness"), 120.0)
    contrast = _safe_float(metrics.get("contrast"), 20.0)
    width = _safe_float(metrics.get("width_px"), 512.0)
    height = _safe_float(metrics.get("height_px"), 512.0)

    blur_score = min(100.0, max(0.0, blur / 40.0 * 100.0))
    brightness_score = max(0.0, 100.0 - abs(brightness - 130.0) / 130.0 * 100.0)
    contrast_score = min(100.0, max(0.0, contrast / 30.0 * 100.0))
    resolution_score = min(100.0, max(0.0, min(width, height) / 512.0 * 100.0))

    score = 0.30 * blur_score + 0.25 * brightness_score + 0.25 * contrast_score + 0.20 * resolution_score
    score = round(float(score), 1)

    reasons = []
    if blur_score < 60:
        reasons.append("possible blur/focus limitation")
    if brightness_score < 60:
        reasons.append("brightness is not ideal")
    if contrast_score < 60:
        reasons.append("contrast is limited")
    if resolution_score < 80:
        reasons.append("resolution is low")

    status = "PASS" if score >= 70 else "REVIEW_NEEDED" if score >= 50 else "RETAKE_RECOMMENDED"
    return {
        "image_quality_score": score,
        "image_quality_level": status,
        "reasons": reasons or ["basic image quality measurements are acceptable"],
    }


def summarize_thermal_measurements(thermal_result: Dict) -> Dict:
    high_mask = thermal_result.get("high_mask") if isinstance(thermal_result, dict) else None
    medium_mask = thermal_result.get("medium_mask") if isinstance(thermal_result, dict) else None
    normalized_map = thermal_result.get("normalized_map") if isinstance(thermal_result, dict) else None

    hot_ratio = _safe_float(thermal_result.get("hot_region_ratio"))
    medium_ratio = _safe_float(thermal_result.get("medium_region_ratio"))
    asymmetry = _safe_float(thermal_result.get("asymmetry_score"))

    dominant_zone = _dominant_mask_zone(high_mask)
    medium_zone = _dominant_mask_zone(medium_mask)

    if normalized_map is not None:
        arr = np.asarray(normalized_map, dtype=np.float32)
        mean_intensity = float(arr.mean()) if arr.size else 0.0
        max_intensity = float(arr.max()) if arr.size else 0.0
        thermal_contrast = float(arr.std()) if arr.size else 0.0
    else:
        mean_intensity = max_intensity = thermal_contrast = 0.0

    concern_points = []
    if hot_ratio >= 0.08:
        concern_points.append("large high-monitoring thermal area")
    if medium_ratio >= 0.15:
        concern_points.append("noticeable medium-monitoring thermal area")
    if asymmetry >= 0.10:
        concern_points.append("high left-right thermal asymmetry")
    elif asymmetry >= 0.05:
        concern_points.append("moderate left-right thermal asymmetry")

    return {
        "hotspot_ratio_percent": _percent(hot_ratio),
        "medium_zone_ratio_percent": _percent(medium_ratio),
        "asymmetry_score": round(float(asymmetry), 4),
        "dominant_high_monitoring_zone": dominant_zone,
        "dominant_medium_monitoring_zone": medium_zone,
        "mean_relative_intensity": round(float(mean_intensity), 4),
        "max_relative_intensity": round(float(max_intensity), 4),
        "thermal_contrast": round(float(thermal_contrast), 4),
        "concern_points": concern_points or ["no strong measurement-based thermal concern beyond the displayed overlay"],
    }


def summarize_ulcer_measurements(ulcer_result: Dict) -> Dict:
    if not isinstance(ulcer_result, dict) or not ulcer_result.get("model_available"):
        return {
            "available": False,
            "detected": None,
            "area_pixels": None,
            "area_percent_of_image": None,
            "largest_region_area_pixels": None,
            "dominant_ulcer_zone": "Not available",
            "shape_irregularity": None,
            "summary_points": ["ulcer segmentation measurements are not available"],
        }

    mask = np.asarray(ulcer_result.get("mask"))
    area = int(ulcer_result.get("total_area_pixels") or 0)
    total_pixels = int(mask.size) if mask is not None and mask.size else 0
    area_percent = round((area / total_pixels * 100.0), 2) if total_pixels else 0.0

    features = ulcer_result.get("features", {}) or {}
    lesions = features.get("lesions", []) or []
    largest = max([int(l.get("area_pixels", 0)) for l in lesions], default=0)

    dominant_zone = _dominant_mask_zone(mask)

    # Simple shape proxy: bounding-box occupancy. Lower values suggest more irregular/spread-out shape.
    shape_irregularity = None
    if mask is not None and mask.sum() > 0:
        ys, xs = np.where(mask > 0)
        bbox_area = max(1, (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        occupancy = area / bbox_area
        shape_irregularity = round(float(1.0 - occupancy), 4)

    points = []
    if area > 0:
        points.append(f"visible ulcer-like area covers {area_percent}% of the analyzed image")
        points.append(f"dominant visible ulcer-like location: {dominant_zone}")
        if shape_irregularity is not None:
            points.append(f"shape irregularity proxy: {shape_irregularity}")
    else:
        points.append("no visible ulcer-like region was detected by the segmentation model")

    return {
        "available": True,
        "detected": bool(ulcer_result.get("visible_ulcer_like_region_detected")),
        "area_pixels": area,
        "area_percent_of_image": area_percent,
        "largest_region_area_pixels": largest,
        "dominant_ulcer_zone": dominant_zone,
        "shape_irregularity": shape_irregularity,
        "summary_points": points,
    }


def compute_followup_change(current_value: Optional[float], previous_value: Optional[float], label: str) -> Dict:
    if previous_value is None or previous_value <= 0 or current_value is None:
        return {"available": False, "label": label, "message": "No previous comparable value was provided."}
    change = float(current_value) - float(previous_value)
    pct = change / float(previous_value) * 100.0
    if pct > 10:
        status = "increase"
    elif pct < -10:
        status = "decrease"
    else:
        status = "stable range"
    return {
        "available": True,
        "label": label,
        "previous": previous_value,
        "current": current_value,
        "change": round(change, 4),
        "change_percent": round(pct, 2),
        "status": status,
        "message": f"{label} shows {status} ({pct:.1f}% change).",
    }


def compute_monitoring_score(
    dm_result: Dict,
    thermal_measurements: Dict,
    ulcer_measurements: Dict,
    rgb_quality_score: Dict,
    previous_area_pixels: Optional[float] = None,
    previous_hotspot_ratio: Optional[float] = None,
) -> Dict:
    score = 0.0
    reasons = []

    dm_prob = _safe_float(dm_result.get("diabetic_foot_pattern_probability"), 0.0)
    if dm_prob >= 0.80:
        score += 25
        reasons.append("RGB+thermal pattern is strongly diabetic-foot-like within the STANDUP-trained model")
    elif dm_prob >= 0.50:
        score += 15
        reasons.append("RGB+thermal pattern leans diabetic-foot-like within the STANDUP-trained model")

    hot_ratio = thermal_measurements.get("hotspot_ratio_percent", 0.0) / 100.0
    asymmetry = thermal_measurements.get("asymmetry_score", 0.0)
    if hot_ratio >= 0.08:
        score += 25
        reasons.append("high thermal hotspot ratio")
    elif hot_ratio >= 0.03:
        score += 15
        reasons.append("moderate thermal hotspot ratio")

    if asymmetry >= 0.10:
        score += 15
        reasons.append("high thermal asymmetry")
    elif asymmetry >= 0.05:
        score += 8
        reasons.append("moderate thermal asymmetry")

    if ulcer_measurements.get("available") and ulcer_measurements.get("detected"):
        area_pct = _safe_float(ulcer_measurements.get("area_percent_of_image"), 0.0)
        if area_pct >= 5.0:
            score += 25
            reasons.append("large visible ulcer-like segmented area")
        elif area_pct >= 1.0:
            score += 18
            reasons.append("visible ulcer-like segmented area")
        else:
            score += 10
            reasons.append("small visible ulcer-like segmented area")

    quality_score = _safe_float(rgb_quality_score.get("image_quality_score"), 70.0)
    if quality_score < 50:
        score += 8
        reasons.append("low image quality increases uncertainty and retake priority")

    area_change = compute_followup_change(
        ulcer_measurements.get("area_pixels") if ulcer_measurements.get("available") else None,
        previous_area_pixels,
        "visible ulcer-like area",
    )
    hotspot_change = compute_followup_change(hot_ratio, previous_hotspot_ratio, "thermal hotspot ratio")

    if area_change.get("available") and area_change.get("change_percent", 0) > 10:
        score += 10
        reasons.append("visible ulcer-like area increased compared with previous visit")
    if hotspot_change.get("available") and hotspot_change.get("change_percent", 0) > 10:
        score += 8
        reasons.append("thermal hotspot ratio increased compared with previous visit")

    score = int(max(0, min(100, round(score))))
    if score >= 70:
        level = "HIGH_MONITORING"
    elif score >= 40:
        level = "MEDIUM_MONITORING"
    else:
        level = "LOW_MONITORING"

    return {
        "monitoring_score": score,
        "monitoring_level": level,
        "main_reasons": reasons or ["no strong measurement-based concern was found"],
        "followup_area_change": area_change,
        "followup_hotspot_change": hotspot_change,
        "safety_note": SAFETY_NOTE,
    }


def build_measurement_summary(rgb_quality, dm_result, thermal_result, ulcer_result, previous_area_pixels=None, previous_hotspot_ratio=None):
    image_quality_score = compute_image_quality_score(rgb_quality)
    thermal_measurements = summarize_thermal_measurements(thermal_result)
    ulcer_measurements = summarize_ulcer_measurements(ulcer_result)
    monitoring_score = compute_monitoring_score(
        dm_result=dm_result,
        thermal_measurements=thermal_measurements,
        ulcer_measurements=ulcer_measurements,
        rgb_quality_score=image_quality_score,
        previous_area_pixels=previous_area_pixels,
        previous_hotspot_ratio=previous_hotspot_ratio,
    )
    return {
        "image_quality_score": image_quality_score,
        "thermal_measurements": thermal_measurements,
        "ulcer_measurements": ulcer_measurements,
        "monitoring_score": monitoring_score,
        "safety_note": SAFETY_NOTE,
    }
