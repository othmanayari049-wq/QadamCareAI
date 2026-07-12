from pathlib import Path
import cv2
import numpy as np


def validate_thermal_image(image_path):
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))

    if img is None:
        return {
            "is_valid": False,
            "status": "FAIL",
            "message": "Image could not be read.",
            "warnings": [],
            "metrics": {},
            "safety_note": "Thermal validation failed."
        }

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(img)

    warnings = []

    if w < 128 or h < 128:
        warnings.append("Image resolution is too low for thermal review.")

    # Natural RGB photos usually have strong channel correlation.
    # Pseudo-colored thermograms often have weaker RGB channel correlation.
    rg_corr = float(np.corrcoef(r.flatten(), g.flatten())[0, 1])
    rb_corr = float(np.corrcoef(r.flatten(), b.flatten())[0, 1])
    gb_corr = float(np.corrcoef(g.flatten(), b.flatten())[0, 1])

    mean_corr = np.nanmean([rg_corr, rb_corr, gb_corr])

    # Thermal pseudo-color images often have unusual color dominance.
    red_dominance = float(np.mean(r) - np.mean(g))
    blue_dominance = float(np.mean(b) - np.mean(g))

    gray_std = float(np.std(gray))
    saturation = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    mean_saturation = float(np.mean(saturation))

    unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))

    # Main RGB rejection logic
    if mean_corr > 0.88 and mean_saturation < 90:
        warnings.append(
            "Image looks like a natural RGB photograph, not a pseudo-colored thermogram."
        )

    if mean_saturation < 45:
        warnings.append(
            "Image has low color saturation; it may not be a thermography image."
        )

    if gray_std < 12:
        warnings.append(
            "Thermal/intensity variation is too low."
        )

    if unique_colors < 50:
        warnings.append(
            "Image has limited color diversity."
        )

    # Accept only if it looks thermal-like enough
    if len(warnings) >= 1:
        status = "REVIEW NEEDED"
        is_valid = False
        message = (
            "Uploaded image does not clearly look like a valid thermography image. "
            "Please upload a real thermal/pseudo-colored thermogram from a thermal camera or approved dataset."
        )
    else:
        status = "PASS"
        is_valid = True
        message = "Image passed basic thermography-format screening."

    return {
        "is_valid": is_valid,
        "status": status,
        "message": message,
        "warnings": warnings,
        "metrics": {
            "width": w,
            "height": h,
            "unique_colors": unique_colors,
            "mean_saturation": round(mean_saturation, 2),
            "intensity_variation": round(gray_std, 2),
            "rgb_channel_correlation": round(float(mean_corr), 4),
            "red_dominance": round(red_dominance, 2),
            "blue_dominance": round(blue_dominance, 2),
        },
        "safety_note": (
            "This validation does not prove medical calibration. "
            "A valid thermal AI model requires images from a known thermal camera or validated thermography dataset."
        ),
    }
