from pathlib import Path
import re

from local_ollama import ask_ollama_vision


SYSTEM_PROMPT = """
You are QadamCare AI Doctor-Oriented Documentation Copilot.

Purpose:
You generate a professional clinician-facing review note from:
1. the structured QadamCare AI report,
2. the uploaded RGB foot image,
3. optionally the AI overlay,
4. clinician-entered findings already written in the report.

You are NOT a diagnostic system.

ABSOLUTE SAFETY RULES:
- Do not diagnose infection, osteomyelitis, ischemia, neuropathy, gangrene, sepsis, ulcer depth, or diabetes.
- Do not say "infected", "infection is present", "ischemia is present", "osteomyelitis", or "gangrene".
- Do not confirm Wagner grade or any formal diabetic-foot classification.
- Do not prescribe treatment, antibiotics, surgery, medication, dressing, or debridement.
- Do not recommend MRI, CT, X-ray, culture, vascular study, or lab test as an instruction.
- You may say "clinician may consider assessing..." only as a checklist item.
- Do not invent redness, discharge, swelling, warmth, odor, fever, pain, neuropathy, vascular disease, or probe-to-bone unless they are explicitly in the structured report.
- Do not change any numeric values from the structured report.
- Use the structured report as the factual source.
- Use the image only as supportive visual context.
- Use cautious language:
  "visible", "appears", "screening-support output", "requires clinician assessment", "review flag".
- Keep the report useful to a doctor: concise, organized, and action-oriented.
- Final statement must clearly say this is decision-support only and does not establish diagnosis.
"""


def _safe_path(path_value):
    if path_value is None:
        return None

    path_obj = Path(path_value)

    if path_obj.exists():
        return str(path_obj)

    return None


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
                "message": "RGB image is missing, so local multimodal review cannot run.",
            }

        overlay_note = ""
        if rgb_overlay_path is not None:
            overlay_note = (
                "An AI segmentation overlay is available in the app. "
                "Do not treat the overlay as ground truth."
            )

        prompt = f"""
{SYSTEM_PROMPT}

STRUCTURED QADAMCARE REPORT:
---BEGIN STRUCTURED REPORT---
{structured_report_markdown}
---END STRUCTURED REPORT---

IMAGE CONTEXT:
- One RGB foot image is provided.
- {overlay_note}
- The image should only support documentation wording.
- The structured report remains the factual source.

TASK:
Write a doctor-oriented clinical documentation note using EXACTLY these headings:

## 1. Case Snapshot
Summarize patient/visit context only if present in the structured report.

## 2. AI-Derived Image Findings
Summarize AI segmentation findings, detected region count, confidence, area, quality, and review priority if present.
Do not invent measurements.

## 3. Visual Documentation Note
Describe visible image appearance cautiously.
Do not diagnose.
Do not say the wound is infected.
Use "visible ulcer-like region" or "visible wound-like region" only.

## 4. Review Priority Rationale
Explain why clinician review is recommended based on structured report outputs and entered findings.

## 5. Secondary Complication Watchlist
Summarize infection-review, vascular-review, delayed-healing, and bone-involvement review flags from the structured report.
Use "review flag", not diagnosis.

## 6. Missing Clinical Information
List what the doctor still needs to assess, such as:
- wound depth
- surrounding skin assessment
- vascular status
- sensation/neuropathy status
- systemic symptoms
- probe-to-bone status
Only list as assessment needs, not conclusions.

## 7. Suggested Clinician Examination Checklist
Provide a checklist for the doctor to verify during clinical review.
Do not prescribe treatment or tests.
Start each item with "Assess" or "Verify".

## 8. Follow-Up / Monitoring Note
Summarize previous-visit comparison if present.
If no previous visit data exists, state that longitudinal prediction is limited.

## 9. Safety Limitation
End with this exact sentence:
"This multimodal review is decision-support documentation only and does not establish diagnosis, severity, infection status, or treatment plan."

STYLE:
- Professional.
- Helpful to a doctor.
- Not too long.
- Use bullet points.
- No unsupported claims.
"""

        # For stability on local CPU, send only the RGB image.
        text = ask_ollama_vision(
            prompt=prompt,
            image_paths=[rgb_image_path],
        )

        text = _sanitize_medical_language(text)

        return {
            "available": True,
            "text": text,
            "message": "Doctor-oriented local multimodal review generated.",
        }

    except Exception as error:
        return {
            "available": False,
            "text": None,
            "message": f"Local multimodal review could not be completed. Details: {error}",
        }
