from pathlib import Path
import tempfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from standup_roi_thermal_analysis import analyze_relative_thermal_with_rois


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        thermal = np.zeros((120, 160), dtype=np.uint8)
        thermal[:, :80] = np.tile(np.linspace(30, 200, 80, dtype=np.uint8), (120, 1))
        thermal[:, 80:] = np.tile(np.linspace(45, 215, 80, dtype=np.uint8), (120, 1))
        thermal_path = tmp / "thermal.png"
        cv2.imwrite(str(thermal_path), thermal)

        whole = analyze_relative_thermal_with_rois(thermal_path)
        assert whole["roi_status"] == "NOT_AVAILABLE"
        assert whole["clinical_monitoring_level"] == "Not determined"
        assert whole["anatomical_asymmetry_available"] is False
        assert 9.0 <= whole["upper_decile_coverage_percent"] <= 11.0

        left = np.zeros_like(thermal)
        right = np.zeros_like(thermal)
        left[10:110, 10:70] = 255
        right[10:110, 90:150] = 255
        left_path = tmp / "left.png"
        right_path = tmp / "right.png"
        cv2.imwrite(str(left_path), left)
        cv2.imwrite(str(right_path), right)

        roi = analyze_relative_thermal_with_rois(thermal_path, left_path, right_path)
        assert roi["roi_status"] == "AVAILABLE"
        assert roi["clinical_monitoring_level"] == "Not determined"
        assert roi["anatomical_asymmetry_available"] is True
        assert set(roi["zone_asymmetry"]) == {"toes/forefoot", "midfoot", "heel"}
        assert roi["overlay"] is not None

        overlap_path = tmp / "overlap.png"
        cv2.imwrite(str(overlap_path), left)
        try:
            analyze_relative_thermal_with_rois(thermal_path, left_path, overlap_path)
        except ValueError as error:
            assert "overlap" in str(error).lower()
        else:
            raise AssertionError("Overlapping masks should be rejected.")

    print("PASS: whole-frame output is descriptive and non-clinical")
    print("PASS: ROI-aware zone asymmetry is generated only with two masks")
    print("PASS: overlapping masks are rejected")
    print("All QadamCare v6 STANDUP feature checks passed.")


if __name__ == "__main__":
    main()
