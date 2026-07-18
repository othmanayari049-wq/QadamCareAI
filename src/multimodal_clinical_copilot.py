from pathlib import Path
import re

from local_ollama import ask_ollama_vision


SYSTEM_PROMPT = """
You are QadamCare AI's clinician-facing visual documentation assistant.

Use the uploaded image only as supportive visual context. The structured report is the factual source.

Rules:
- Do not diagnose diabetes, infection, ischemia, neuropathy, osteomyelitis, gangrene, sepsis, ulcer depth, or severity.
- Do not prescribe treatment, tests, medication, surgery, antibiotics, dressing, or debridement.
- Do not invent symptoms, measurements, or history.
- Preserve all supplied numbers.
- Use cautious language such as: visible, appears, review flag, requires clinician assessment.
- Keep the note concise.
"""


def _safe_path(path_value):
    if path_value is None:
        return None
    path_obj = Path(path_value)
    return str(path_obj) if path_obj.exists() else None


def _sanitize_medical_language(text):
    replacements = {
        "possibly infected": "requiring clinician assessment for possible infection-related concern",
        "appears infected": "requires clinician assessment for infection-related concern",
        "signs of infection": "features that may require clinician infection review",
        "visible signs of infection": "visible features that may require clinician infection review",
        "infected": "requiring clinician assessment",
        "infection is present": "infection cannot be confirmed by this prototype",
        "osteomyelitis": "bone-involvement concern requiring clinician assessment",
        "gangrene": "severe tissue concern requiring clinician assessment",
        "ischemia is present": "vascular status cannot be confirmed by this prototype",
    }
    for old, new in replacements.items():
        text = re.sub(old, new, text, flags=re.IGNORECASE)
    return text.strip()


def _compact_report(report, max_chars=5000):
    lines = []
    for raw_line in str(report).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        lines.append(line)

    compact = "\n".join(lines)
    if len(compact) > max_chars:
        compact = compact[:max_chars] + "\n[Report shortened for local VLM context.]"
    return compact


def build_multimodal_summary(
    structured_report_markdown,
    rgb_image_path,
    rgb_overlay_path,
    thermal_image_path=None,
    thermal_attention_path=None,
):
    try:
        rgb_image_path = _safe_path(rgb_image_path)
        rgb_overlay_path = _safe_path(rgb_overlay_path)

        if rgb_image_path is None:
            return {
                "available": False,
                "text": None,
                "message": "An image is missing, so the local visual documentation note cannot run.",
            }

        report = _compact_report(structured_report_markdown)
        overlay_note = (
            "A segmentation/attention overlay exists in the application; it is not ground truth."
            if rgb_overlay_path
            else "No valid AI overlay is available."
        )

        prompt = f"""
{SYSTEM_PROMPT}

STRUCTURED CASE DATA:
{report}

IMAGE CONTEXT:
- One workflow-compatible image is attached.
- {overlay_note}

Write a concise note with exactly these headings:

## Case Context
Include age, gender, diabetes type, workflow, and entered symptoms only when supplied.

## Visual Documentation
Describe only broad visible appearance. Do not infer infection, depth, circulation, or severity.

## Model-Supported Findings
Summarize only completed and valid model outputs from the structured data.

## Clinical Review Priorities
Explain which entered findings or valid outputs make clinician review more important.

## Missing Information
List the main clinical information still needed before medical conclusions can be made.

## Safety Limitation
End exactly with:
This multimodal review is decision-support documentation only and does not establish diagnosis, severity, infection status, or treatment plan.
"""

        text = ask_ollama_vision(prompt=prompt, image_paths=[rgb_image_path])
        text = _sanitize_medical_language(text)

        return {
            "available": True,
            "text": text,
            "message": "Concise local visual documentation note generated.",
        }

    except Exception as error:
        return {
            "available": False,
            "text": None,
            "message": f"Local multimodal review could not be completed. Details: {error}",
        }
