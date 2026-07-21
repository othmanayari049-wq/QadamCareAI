"""Verify QadamCare v3 measurement and tri-state logic without model checkpoints."""

from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from advanced_wound_measurements import calculate_advanced_wound_measurements
from secondary_complication_engine import analyze_complication_pathways


def verify_measurements():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        image[:] = (90, 120, 150)
        image_path = temp / "image.png"
        cv2.imwrite(str(image_path), image)

        model_mask = np.zeros((256, 256), dtype=np.uint8)
        model_mask[80:160, 90:170] = 1

        foot_mask = np.zeros((200, 300), dtype=np.uint8)
        foot_mask[20:190, 60:250] = 255
        foot_mask_path = temp / "foot_mask.png"
        cv2.imwrite(str(foot_mask_path), foot_mask)

        result = calculate_advanced_wound_measurements(
            segmentation_mask=model_mask,
            original_image_path=image_path,
            foot_roi_mask_path=foot_mask_path,
            reference_length_cm=2.0,
            reference_length_pixels=40.0,
        )
        assert result["wound_area_pixels_original"] > 0
        assert result["foot_relative_area_percent"] is not None
        assert result["estimated_area_cm2"] is not None
        print("PASS: advanced wound measurements")
        print(result)


def verify_tri_state_logic():
    unknown_inputs = {
        "pain_assessed": False,
        "pain_level": None,
        "redness": None,
        "swelling": None,
        "warmth": None,
        "discharge": None,
        "fever": None,
        "neuropathy": None,
        "vascular_disease": None,
        "probe_to_bone": None,
        "periwound_callus": None,
        "undermining_tunneling": None,
    }
    unknown = analyze_complication_pathways(
        rgb_result={"area_pixels": 1000},
        clinical_inputs=unknown_inputs,
    )
    assert unknown["escalation_priority"] == "Insufficient assessed information"
    assert unknown["unassessed_field_count"] > 0

    assessed_negative = dict(unknown_inputs)
    assessed_negative.update(
        {
            "pain_assessed": True,
            "pain_level": 0,
            "redness": False,
            "swelling": False,
            "warmth": False,
            "discharge": False,
            "fever": False,
            "neuropathy": False,
            "vascular_disease": False,
            "probe_to_bone": False,
            "periwound_callus": False,
            "undermining_tunneling": False,
        }
    )
    negative = analyze_complication_pathways(
        rgb_result={"area_pixels": 1000},
        clinical_inputs=assessed_negative,
    )
    assert negative["escalation_priority"] == "No escalation generated from the assessed entries"
    assert "absent" in negative["reasons"][0].lower() or negative["unassessed_field_count"] == 0

    positive = dict(assessed_negative)
    positive.update({"redness": True, "warmth": True, "discharge": True, "fever": True})
    flagged = analyze_complication_pathways(
        rgb_result={"area_pixels": 1000},
        clinical_inputs=positive,
    )
    assert flagged["infection_review_flag"] in {"Moderate", "High"}
    print("PASS: tri-state complication logic")


if __name__ == "__main__":
    verify_measurements()
    verify_tri_state_logic()
    print("All QadamCare v3 feature checks passed.")
