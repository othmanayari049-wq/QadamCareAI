from pathlib import Path
import cv2
import numpy as np


def analyze_thermal_image(image_path):
    image_path = Path(image_path)

    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "thermal_available": False,
            "thermal_concern": "NOT AVAILABLE",
            "hot_region_ratio": None,
            "mean_intensity": None,
            "max_intensity": None,
            "note": "Thermal image could not be read.",
            "reasons": [],
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_intensity = float(np.mean(gray))
    max_intensity = float(np.max(gray))

    threshold = np.percentile(gray, 90)
    hot_region = gray >= threshold
    hot_region_ratio = float(hot_region.sum() / hot_region.size)

    reasons = []

    if hot_region_ratio > 0.20:
        concern = "HIGH"
        reasons.append("Large high-temperature region detected in the thermogram.")
    elif hot_region_ratio > 0.10:
        concern = "MODERATE"
        reasons.append("Moderate high-temperature region detected in the thermogram.")
    else:
        concern = "LOW"
        reasons.append("No large high-temperature region detected by the prototype.")

    if max_intensity > 230:
        reasons.append("Very high thermal intensity values are present.")

    return {
        "thermal_available": True,
        "thermal_concern": concern,
        "hot_region_ratio": round(hot_region_ratio, 4),
        "mean_intensity": round(mean_intensity, 2),
        "max_intensity": round(max_intensity, 2),
        "note": (
            "Thermal analysis is a prototype screening-support feature. "
            "It does not measure calibrated medical temperature unless the thermal camera provides calibrated values."
        ),
        "reasons": reasons,
    }
