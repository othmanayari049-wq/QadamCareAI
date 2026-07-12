from pathlib import Path
import cv2
import numpy as np


def assess_image_quality(
    image_path,
    min_width=512,
    min_height=512,
    min_blur_score=25.0,
    min_brightness=35.0,
    max_brightness=225.0,
):
    """
    Technical image-quality screening for the prototype.

    This checks resolution, blur, brightness, clipping, and contrast.
    It does NOT yet confirm that the whole foot is visible or that the
    camera angle is clinically correct.
    """
    image_path = Path(image_path)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    dark_ratio = float(np.mean(gray <= 15))
    bright_ratio = float(np.mean(gray >= 240))

    blockers = []
    warnings = []

    if width < min_width or height < min_height:
        blockers.append("Image resolution is too low.")

    if blur_score < min_blur_score:
        blockers.append("Image appears blurry.")

    if brightness < min_brightness:
        blockers.append("Image is too dark.")

    if brightness > max_brightness:
        blockers.append("Image is too bright.")

    if dark_ratio > 0.35:
        warnings.append("Large dark area detected; improve lighting.")

    if bright_ratio > 0.35:
        warnings.append("Large bright area detected; reduce glare.")

    if contrast < 12:
        warnings.append("Low contrast detected.")

    status = "PASS" if not blockers else "RETAKE IMAGE"

    return {
        "image_name": image_path.name,
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "metrics": {
            "width_px": width,
            "height_px": height,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "blur_score": round(blur_score, 2),
            "dark_pixel_ratio": round(dark_ratio, 4),
            "bright_pixel_ratio": round(bright_ratio, 4),
        },
    }


def print_quality_report(result):
    print("\nImage Quality Report")
    print("=" * 45)
    print("Image :", result["image_name"])
    print("Status:", result["status"])

    print("\nMetrics:")
    for key, value in result["metrics"].items():
        print(f"- {key}: {value}")

    if result["blockers"]:
        print("\nRetake reasons:")
        for item in result["blockers"]:
            print(f"- {item}")

    if result["warnings"]:
        print("\nWarnings:")
        for item in result["warnings"]:
            print(f"- {item}")