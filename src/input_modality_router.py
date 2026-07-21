from pathlib import Path

import cv2
import numpy as np


THERMAL_STANDUP_GRAYSCALE = "standup_grayscale_thermal"
THERMAL_PSEUDOCOLOR = "pseudocolor_thermal"
NATURAL_RGB = "natural_rgb"
UNKNOWN_IMAGE = "unknown"


def _safe_corr(a, b):
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    if float(a.std()) < 1e-6 or float(b.std()) < 1e-6:
        return 1.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 1.0


def inspect_image(image_path):
    """Return format measurements without making a medical interpretation."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "readable": False,
            "image_type": UNKNOWN_IMAGE,
            "message": f"Image could not be read: {image_path}",
            "metrics": {},
        }

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image)

    mean_saturation = float(hsv[:, :, 1].mean())
    intensity_std = float(gray.std())
    mean_brightness = float(gray.mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    channel_difference = float(
        (
            np.mean(np.abs(red.astype(np.float32) - green.astype(np.float32)))
            + np.mean(np.abs(red.astype(np.float32) - blue.astype(np.float32)))
            + np.mean(np.abs(green.astype(np.float32) - blue.astype(np.float32)))
        )
        / 3.0
    )
    correlations = [
        _safe_corr(red, green),
        _safe_corr(red, blue),
        _safe_corr(green, blue),
    ]
    mean_channel_correlation = float(np.mean(correlations))

    # STANDUP thermal PNGs are monochrome images stored in three identical channels.
    is_monochrome = mean_saturation <= 12 and channel_difference <= 4.0

    # Pseudo-coloured plantar thermograms contain deliberately mapped colours.
    is_pseudocolor = (
        mean_saturation >= 65
        and channel_difference >= 18
        and mean_channel_correlation <= 0.96
    )

    if is_monochrome:
        image_type = THERMAL_STANDUP_GRAYSCALE
    elif is_pseudocolor:
        image_type = THERMAL_PSEUDOCOLOR
    else:
        image_type = NATURAL_RGB

    return {
        "readable": True,
        "image_type": image_type,
        "message": "Image format inspected successfully.",
        "metrics": {
            "width": int(width),
            "height": int(height),
            "mean_saturation": round(mean_saturation, 2),
            "mean_brightness": round(mean_brightness, 2),
            "intensity_variation": round(intensity_std, 2),
            "blur_score": round(blur_score, 2),
            "mean_channel_difference": round(channel_difference, 2),
            "mean_channel_correlation": round(mean_channel_correlation, 4),
        },
    }


def validate_standup_thermal(image_path):
    inspection = inspect_image(image_path)
    warnings = []

    if not inspection["readable"]:
        return {
            "is_valid": False,
            "status": "FAIL",
            "image_type": UNKNOWN_IMAGE,
            "message": inspection["message"],
            "warnings": [],
            "metrics": {},
        }

    metrics = inspection["metrics"]
    if inspection["image_type"] != THERMAL_STANDUP_GRAYSCALE:
        warnings.append("This is not a STANDUP-style monochrome thermal image.")
    if metrics["width"] < 128 or metrics["height"] < 128:
        warnings.append("Image resolution is too low.")
    if metrics["intensity_variation"] < 12:
        warnings.append("Thermal intensity variation is too low.")

    is_valid = len(warnings) == 0
    return {
        "is_valid": is_valid,
        "status": "PASS" if is_valid else "REVIEW NEEDED",
        "image_type": inspection["image_type"],
        "message": (
            "Valid STANDUP-style monochrome thermal input. Use only with its matching plantar RGB image."
            if is_valid
            else "The image does not satisfy the STANDUP paired-input contract."
        ),
        "warnings": warnings,
        "metrics": metrics,
        "recommended_pipeline": "STANDUP RGB + thermal fusion",
        "safety_note": "Format validation does not prove camera calibration or clinical validity.",
    }


def validate_pseudocolor_thermal(image_path):
    inspection = inspect_image(image_path)
    warnings = []

    if not inspection["readable"]:
        return {
            "is_valid": False,
            "status": "FAIL",
            "image_type": UNKNOWN_IMAGE,
            "message": inspection["message"],
            "warnings": [],
            "metrics": {},
        }

    metrics = inspection["metrics"]
    if inspection["image_type"] != THERMAL_PSEUDOCOLOR:
        warnings.append("This is not a pseudo-coloured plantar thermogram.")
    if metrics["width"] < 128 or metrics["height"] < 128:
        warnings.append("Image resolution is too low.")
    if metrics["intensity_variation"] < 12:
        warnings.append("Thermal intensity variation is too low.")

    is_valid = len(warnings) == 0
    return {
        "is_valid": is_valid,
        "status": "PASS" if is_valid else "REVIEW NEEDED",
        "image_type": inspection["image_type"],
        "message": (
            "Valid pseudo-coloured plantar thermogram for the legacy thermal-only research model."
            if is_valid
            else "The image does not satisfy the pseudo-colour thermal input contract."
        ),
        "warnings": warnings,
        "metrics": metrics,
        "recommended_pipeline": "Pseudo-colour thermal-only classifier / attention map",
        "safety_note": "Colour values are display intensities unless calibrated raw temperature data are available.",
    }


def detect_faces(image_path):
    """Best-effort face gate that never crashes the analysis workflow.

    Some broken or conflicting OpenCV installations expose basic image operations but
    omit CascadeClassifier or cv2.data. In that case the face check is marked unavailable
    and the remaining validation continues instead of failing the whole analysis.
    """
    result = {
        "faces": [],
        "available": False,
        "warning": None,
    }

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        result["warning"] = "Face-screening check was unavailable because the image could not be read."
        return result

    cascade_class = getattr(cv2, "CascadeClassifier", None)
    cv2_data = getattr(cv2, "data", None)
    haar_root = getattr(cv2_data, "haarcascades", None) if cv2_data is not None else None
    if cascade_class is None or not haar_root:
        result["warning"] = (
            "Optional face-screening check was skipped because this OpenCV installation "
            "does not include the Haar-cascade components."
        )
        return result

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade_path = str(Path(haar_root) / "haarcascade_frontalface_default.xml")
        detector = cascade_class(cascade_path)
        if detector.empty():
            result["warning"] = "Optional face-screening model could not be loaded."
            return result

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        result["faces"] = [tuple(int(value) for value in face) for face in faces]
        result["available"] = True
        return result
    except Exception as error:
        result["warning"] = f"Optional face-screening check was skipped: {error}"
        return result


def validate_wound_rgb(image_path):
    """Applicability gate for the FUSeg close-up wound segmentation model."""
    inspection = inspect_image(image_path)
    if not inspection["readable"]:
        return {
            "is_valid": False,
            "status": "FAIL",
            "message": "RGB image could not be read.",
            "warnings": [],
            "metrics": {},
        }

    metrics = inspection["metrics"]
    warnings = []
    advisory_notes = []
    face_result = detect_faces(image_path)
    faces = face_result["faces"]

    if inspection["image_type"] != NATURAL_RGB:
        warnings.append(
            "The uploaded file does not look like a normal colour RGB photograph. Use a close-up visible-light foot/wound image."
        )
    if metrics["width"] < 256 or metrics["height"] < 256:
        warnings.append("Image resolution is too low for the FUSeg segmentation model.")
    if metrics["blur_score"] < 25:
        warnings.append("The image appears too blurred for reliable segmentation. Retake it in focus.")
    if not 35 <= metrics["mean_brightness"] <= 225:
        warnings.append("Image brightness is outside the preferred range. Retake it with more even lighting.")
    if metrics["intensity_variation"] < 12:
        warnings.append("Image contrast is too low for reliable segmentation.")
    if faces:
        warnings.append(
            "A face/person scene was detected. FUSeg accepts a close-up foot/wound image, not a full-person photograph."
        )
    if face_result["warning"]:
        advisory_notes.append(face_result["warning"])

    is_valid = len(warnings) == 0
    return {
        "is_valid": is_valid,
        "status": "PASS" if is_valid else "RETAKE / WRONG INPUT",
        "message": (
            "RGB image passed the basic FUSeg applicability and quality gate."
            if is_valid
            else "Ulcer segmentation was blocked because the image did not satisfy the visible-image input contract."
        ),
        "warnings": warnings + advisory_notes,
        "blocking_warnings": warnings,
        "advisory_notes": advisory_notes,
        "metrics": {
            **metrics,
            "faces_detected": len(faces),
            "face_check_available": face_result["available"],
        },
        "recommended_pipeline": "FUSeg close-up RGB wound/ulcer-like segmentation",
        "safety_note": (
            "This gate reduces obvious misuse but does not prove that the image matches the training distribution or that the model output is clinically valid."
        ),
    }


def validate_standup_pair(rgb_path, thermal_path):
    thermal_validation = validate_standup_thermal(thermal_path)
    rgb_inspection = inspect_image(rgb_path)
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    thermal = cv2.imread(str(thermal_path), cv2.IMREAD_COLOR)
    warnings = list(thermal_validation.get("warnings", []))

    if rgb is None:
        warnings.append("Matching RGB image could not be read.")
    elif rgb_inspection.get("image_type") != NATURAL_RGB:
        warnings.append("The first STANDUP input does not look like a normal plantar RGB photograph.")
    if thermal is None:
        warnings.append("Matching thermal image could not be read.")

    if rgb is not None and thermal is not None:
        rgb_ratio = rgb.shape[1] / max(rgb.shape[0], 1)
        thermal_ratio = thermal.shape[1] / max(thermal.shape[0], 1)
        if abs(rgb_ratio - thermal_ratio) > 0.25:
            warnings.append(
                "RGB and thermal aspect ratios differ substantially; confirm that they are a true matched pair."
            )

    is_valid = len(warnings) == 0
    return {
        "is_valid": is_valid,
        "status": "PASS" if is_valid else "REVIEW NEEDED",
        "message": (
            "The pair passed basic STANDUP input checks."
            if is_valid
            else "The STANDUP RGB/thermal pair should not be sent to the fusion model yet."
        ),
        "warnings": warnings,
        "thermal_validation": thermal_validation,
        "rgb_validation": rgb_inspection,
        "recommended_pipeline": "STANDUP RGB + monochrome thermal fusion",
        "safety_note": (
            "Filename or visual checks cannot prove that two images belong to the same participant and capture time."
        ),
    }
