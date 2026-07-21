"""QadamCare professional entry point with a safer visible-image gate.

The legacy Haar face detector is retained only as an advisory signal because it can
mistake toes, wounds, dressings, or skin texture for a face. A face-like detection
must never be the sole reason a clinically relevant close-up foot image is blocked.
All other readability, modality, resolution, blur, brightness, and contrast checks
remain active.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import input_modality_router as _router

_original_validate_wound_rgb = _router.validate_wound_rgb
_FACE_WARNING_PREFIX = "A face/person scene was detected."


def _validate_wound_rgb_without_false_face_block(image_path):
    """Reclassify Haar face-like detections as advisory, never as sole blockers."""
    result = _original_validate_wound_rgb(image_path)

    blocking = list(result.get("blocking_warnings", result.get("warnings", [])))
    existing_advisories = list(result.get("advisory_notes", []))

    face_warnings = [
        warning for warning in blocking if str(warning).startswith(_FACE_WARNING_PREFIX)
    ]
    true_blockers = [
        warning for warning in blocking if not str(warning).startswith(_FACE_WARNING_PREFIX)
    ]

    if face_warnings:
        existing_advisories.append(
            "The optional Haar detector found a face-like pattern, but this heuristic can "
            "misidentify toes, wounds, dressings, or skin texture. It was not used as a "
            "blocking decision. Confirm manually that the upload is a close-up foot/wound image."
        )

    result["blocking_warnings"] = true_blockers
    result["advisory_notes"] = existing_advisories
    result["warnings"] = true_blockers + existing_advisories
    result["is_valid"] = len(true_blockers) == 0
    result["status"] = "PASS" if result["is_valid"] else "RETAKE / WRONG INPUT"
    result["message"] = (
        "RGB image passed the FUSeg applicability and quality gate. Optional face-like "
        "detections are advisory only and require manual confirmation."
        if result["is_valid"]
        else "Ulcer segmentation was blocked because one or more verified image-quality "
        "or input-contract checks failed."
    )
    result.setdefault("metrics", {})["face_detection_is_advisory"] = True
    return result


_router.validate_wound_rgb = _validate_wound_rgb_without_false_face_block

# Execute the existing professional application after installing the safer validator.
_ENTRY = ROOT / "app_qadamcare_pro.py"
_namespace = {"__file__": str(_ENTRY), "__name__": "__main__"}
exec(compile(_ENTRY.read_text(encoding="utf-8"), str(_ENTRY), "exec"), _namespace)
