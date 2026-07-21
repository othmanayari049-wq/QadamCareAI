from pathlib import Path

import cv2
import numpy as np


def _binary_mask(mask):
    array = np.asarray(mask)
    if array.ndim == 3:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return (array > 0).astype(np.uint8)


def _resize_mask(mask, width, height):
    return cv2.resize(
        _binary_mask(mask),
        (int(width), int(height)),
        interpolation=cv2.INTER_NEAREST,
    )


def calculate_advanced_wound_measurements(
    *,
    segmentation_mask,
    original_image_path,
    foot_roi_mask_path=None,
    reference_length_cm=None,
    reference_length_pixels=None,
):
    """Calculate transparent image-relative and optional calibrated measurements.

    The segmentation model works on a resized mask. This function maps that mask back to
    the original uploaded image before calculating original-resolution pixel area.

    Physical area is estimated only when the user supplies a known reference length in
    centimetres and its measured length in pixels on the original image. The estimate
    assumes the reference and wound are in the same plane with limited perspective error.

    Foot-relative area is calculated only from an explicitly supplied binary foot ROI mask;
    no unvalidated automatic foot segmentation is invented.
    """
    image = cv2.imread(str(original_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The original RGB image could not be read for measurement.")

    height, width = image.shape[:2]
    wound_mask_original = _resize_mask(segmentation_mask, width, height)
    wound_pixels_original = int(wound_mask_original.sum())
    image_pixels = int(width * height)

    result = {
        "original_width_px": int(width),
        "original_height_px": int(height),
        "wound_area_pixels_original": wound_pixels_original,
        "area_percent_of_original_image": round(
            100.0 * wound_pixels_original / max(image_pixels, 1), 3
        ),
        "foot_relative_area_percent": None,
        "visible_foot_area_pixels": None,
        "estimated_area_cm2": None,
        "cm_per_pixel": None,
        "measurement_notes": [],
    }

    if foot_roi_mask_path:
        foot_mask_image = cv2.imread(str(Path(foot_roi_mask_path)), cv2.IMREAD_GRAYSCALE)
        if foot_mask_image is None:
            result["measurement_notes"].append(
                "Foot-relative area was unavailable because the uploaded foot ROI mask could not be read."
            )
        else:
            foot_mask = _resize_mask(foot_mask_image, width, height)
            visible_foot_pixels = int(foot_mask.sum())
            wound_inside_foot = int(np.logical_and(wound_mask_original > 0, foot_mask > 0).sum())
            if visible_foot_pixels <= 0:
                result["measurement_notes"].append(
                    "Foot-relative area was unavailable because the foot ROI mask contained no foreground pixels."
                )
            else:
                result["visible_foot_area_pixels"] = visible_foot_pixels
                result["foot_relative_area_percent"] = round(
                    100.0 * wound_inside_foot / visible_foot_pixels, 3
                )
                outside = wound_pixels_original - wound_inside_foot
                if outside > 0:
                    result["measurement_notes"].append(
                        "Part of the predicted wound mask fell outside the supplied foot ROI and was excluded from the foot-relative ratio."
                    )
                result["measurement_notes"].append(
                    "Foot-relative area depends on the accuracy of the user-supplied binary foot ROI mask."
                )
    else:
        result["measurement_notes"].append(
            "Foot-relative area was not calculated because no binary foot ROI mask was supplied."
        )

    try:
        length_cm = float(reference_length_cm or 0)
        length_px = float(reference_length_pixels or 0)
    except (TypeError, ValueError):
        length_cm = 0.0
        length_px = 0.0

    if length_cm > 0 and length_px > 0:
        cm_per_pixel = length_cm / length_px
        result["cm_per_pixel"] = round(cm_per_pixel, 6)
        result["estimated_area_cm2"] = round(
            wound_pixels_original * (cm_per_pixel ** 2), 3
        )
        result["measurement_notes"].append(
            "The cm² value is an approximate planar estimate. It is valid only when the reference object and wound are in the same plane and the reference pixel length was measured on the original uploaded image."
        )
    else:
        result["measurement_notes"].append(
            "Physical area in cm² was not calculated because both a known reference length and its original-image pixel length were not supplied."
        )

    return result
