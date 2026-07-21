"""QadamCare professional app with safer clinical inputs and wound measurements.

Run with:
    python -m streamlit run app_qadamcare_pro_v3.py

This entry point patches the existing unified application at runtime while preserving the
trained model code. It adds tri-state clinical fields, optional foot-relative area, optional
reference-calibrated cm² estimation, and safer rule-based wording.
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import input_modality_router as _router


# Haar face detections are advisory because toes and wound texture can trigger false hits.
_original_validate_wound_rgb = _router.validate_wound_rgb
_FACE_WARNING_PREFIX = "A face/person scene was detected."


def _validate_wound_rgb_without_false_face_block(image_path):
    result = _original_validate_wound_rgb(image_path)
    blocking = list(result.get("blocking_warnings", result.get("warnings", [])))
    advisories = list(result.get("advisory_notes", []))
    face_warnings = [item for item in blocking if str(item).startswith(_FACE_WARNING_PREFIX)]
    true_blockers = [item for item in blocking if not str(item).startswith(_FACE_WARNING_PREFIX)]
    if face_warnings:
        advisories.append(
            "The optional Haar detector found a face-like pattern, but this heuristic can "
            "misidentify toes, wounds, dressings, or skin texture. It was not used as a "
            "blocking decision. Confirm manually that the upload is a close-up foot/wound image."
        )
    result["blocking_warnings"] = true_blockers
    result["advisory_notes"] = advisories
    result["warnings"] = true_blockers + advisories
    result["is_valid"] = not true_blockers
    result["status"] = "PASS" if result["is_valid"] else "RETAKE / WRONG INPUT"
    result["message"] = (
        "RGB image passed the FUSeg applicability and quality gate. Optional face-like "
        "detections are advisory only and require manual confirmation."
        if result["is_valid"]
        else "Ulcer segmentation was blocked because one or more verified image-quality or input-contract checks failed."
    )
    result.setdefault("metrics", {})["face_detection_is_advisory"] = True
    return result


_router.validate_wound_rgb = _validate_wound_rgb_without_false_face_block

BASE_APP = ROOT / "app_unified.py"
source = BASE_APP.read_text(encoding="utf-8")

# Add advanced measurement helper.
source = source.replace(
    "from unified_pdf_report import generate_unified_pdf_report",
    "from unified_pdf_report import generate_unified_pdf_report\nfrom advanced_wound_measurements import calculate_advanced_wound_measurements",
    1,
)

# Keep diabetes history separate from image-model output.
source = source.replace(
    '''    diabetes_type = st.selectbox(
        "Diabetes Type",
        ["Type II", "Type I", "Gestational", "Not specified"],
    )''',
    '''    diabetes_type = st.selectbox(
        "Known diabetes history (user-entered)",
        [
            "Not specified",
            "No known diabetes",
            "Type 1 diabetes",
            "Type 2 diabetes",
            "Gestational diabetes",
            "Other / uncertain",
        ],
        index=0,
        help="User/clinician-entered history; never inferred from the images.",
    )
    st.caption("Diabetes history is user-entered. The image model cannot diagnose diabetes.")''',
    1,
)
source = source.replace(
    'f"- Diabetes type: {patient[\'diabetes_type\']}",',
    'f"- User-entered diabetes history: {patient[\'diabetes_type\']}",',
    1,
)

# Replace Boolean checkboxes with explicit tri-state assessments and add documentation fields.
clinical_widget_pattern = re.compile(
    r'    st\.header\("Clinical Findings"\).*?    probe_to_bone = st\.checkbox\("Positive probe-to-bone finding"\)',
    re.DOTALL,
)
clinical_widgets = '''    st.header("Clinical Findings")
    st.caption("Choose Not assessed when a finding was not examined. It will not be converted to No.")

    pain_assessed = st.checkbox("Pain assessed", value=False)
    pain_level = st.slider("Pain level", 0, 10, 0, disabled=not pain_assessed)

    tri_options = ["Not assessed", "No", "Yes"]
    redness = st.selectbox("Redness", tri_options, index=0)
    swelling = st.selectbox("Swelling", tri_options, index=0)
    warmth = st.selectbox("Warmth", tri_options, index=0)
    discharge = st.selectbox("Discharge or odour", tri_options, index=0)
    fever = st.selectbox("Fever / systemic symptoms", tri_options, index=0)
    neuropathy = st.selectbox("Known or suspected neuropathy", tri_options, index=0)
    vascular_disease = st.selectbox("Known or suspected vascular disease", tri_options, index=0)
    probe_to_bone = st.selectbox("Probe-to-bone finding", tri_options, index=0)

    st.subheader("Additional wound documentation")
    wound_duration = st.selectbox(
        "Wound duration",
        ["Not assessed", "Less than 1 week", "1–4 weeks", "More than 4 weeks"],
        index=0,
    )
    drainage_amount = st.selectbox(
        "Drainage amount",
        ["Not assessed", "None", "Scant", "Moderate", "Heavy"],
        index=0,
    )
    tissue_appearance = st.selectbox(
        "Visible tissue appearance (user-entered)",
        ["Not assessed", "Granulation-like", "Slough-like", "Eschar-like", "Mixed", "Other / uncertain"],
        index=0,
        help="Documentation only; this is not inferred or graded by the segmentation model.",
    )
    periwound_callus = st.selectbox("Periwound callus", tri_options, index=0)
    undermining_tunneling = st.selectbox("Undermining or tunnelling", tri_options, index=0)'''
source, count = clinical_widget_pattern.subn(clinical_widgets, source, count=1)
if count != 1:
    raise RuntimeError("Could not install tri-state clinical input widgets.")

# Add optional measurement inputs to the follow-up section.
followup_pattern = re.compile(
    r'    st\.header\("Follow-Up"\).*?    previous_area_pixels = st\.number_input\(.*?    \)\n',
    re.DOTALL,
)
followup_widgets = '''    st.header("Follow-Up and Measurement")
    previous_area_pixels = st.number_input(
        "Previous comparable wound-like area (pixels)",
        min_value=0,
        value=0,
        step=100,
        help="Use only when the previous image was captured and processed using a comparable protocol.",
    )

    st.subheader("Optional physical calibration")
    reference_length_cm = st.number_input(
        "Known reference length (cm)", min_value=0.0, value=0.0, step=0.1,
        help="Length of a ruler/marker placed in the same plane as the wound. Enter 0 when unavailable.",
    )
    reference_length_pixels = st.number_input(
        "Reference length in original-image pixels", min_value=0.0, value=0.0, step=1.0,
        help="Measure the same reference length on the original uploaded image. Both fields are required for cm² estimation.",
    )
    foot_roi_mask_file = st.file_uploader(
        "Optional binary visible-foot ROI mask",
        type=["png", "jpg", "jpeg"],
        key="foot_roi_mask",
        help="White = visible foot, black = background. Used only for wound-to-visible-foot percentage.",
    )
'''
source, count = followup_pattern.subn(followup_widgets, source, count=1)
if count != 1:
    raise RuntimeError("Could not install optional measurement inputs.")

# Replace clinical data collection and report lines.
functions_pattern = re.compile(
    r'def build_clinical_inputs\(\):.*?def show_validation\(validation, title\):',
    re.DOTALL,
)
functions_replacement = '''def _tri_state(value):
    if value == "Yes":
        return True
    if value == "No":
        return False
    return None


def _display_state(value):
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Not assessed"


def build_clinical_inputs():
    return {
        "pain_assessed": pain_assessed,
        "pain_level": pain_level if pain_assessed else None,
        "redness": _tri_state(redness),
        "swelling": _tri_state(swelling),
        "warmth": _tri_state(warmth),
        "discharge": _tri_state(discharge),
        "fever": _tri_state(fever),
        "neuropathy": _tri_state(neuropathy),
        "vascular_disease": _tri_state(vascular_disease),
        "probe_to_bone": _tri_state(probe_to_bone),
        "periwound_callus": _tri_state(periwound_callus),
        "undermining_tunneling": _tri_state(undermining_tunneling),
        "wound_duration": wound_duration,
        "drainage_amount": drainage_amount,
        "tissue_appearance": tissue_appearance,
    }


def clinical_lines(inputs):
    pain_text = f"{inputs['pain_level']}/10" if inputs.get("pain_assessed") else "Not assessed"
    return [
        f"Pain: {pain_text}",
        f"Redness: {_display_state(inputs['redness'])}",
        f"Swelling: {_display_state(inputs['swelling'])}",
        f"Warmth: {_display_state(inputs['warmth'])}",
        f"Discharge or odour: {_display_state(inputs['discharge'])}",
        f"Fever/systemic symptoms: {_display_state(inputs['fever'])}",
        f"Neuropathy: {_display_state(inputs['neuropathy'])}",
        f"Vascular disease: {_display_state(inputs['vascular_disease'])}",
        f"Probe-to-bone finding: {_display_state(inputs['probe_to_bone'])}",
        f"Periwound callus: {_display_state(inputs['periwound_callus'])}",
        f"Undermining/tunnelling: {_display_state(inputs['undermining_tunneling'])}",
        f"Wound duration: {inputs['wound_duration']}",
        f"Drainage amount: {inputs['drainage_amount']}",
        f"Visible tissue appearance (user-entered): {inputs['tissue_appearance']}",
        "These characteristics are user-entered documentation. The FUSeg model does not infer wound depth, tissue type, infection, ischemia, or severity.",
    ]


def show_validation(validation, title):'''
source, count = functions_pattern.subn(functions_replacement, source, count=1)
if count != 1:
    raise RuntimeError("Could not install tri-state clinical data handling.")

# Replace the ulcer execution block with calibrated and foot-relative measurements.
ulcer_block_pattern = re.compile(
    r'                ulcer = analyze_ulcer_segmentation\(image_path\).*?                sections\.append\(\n                    \{\n                        "heading": "Rule-based complication review pathways".*?                    \}\n                \)\n\n        elif workflow == WORKFLOW_STANDUP:',
    re.DOTALL,
)
ulcer_block = '''                ulcer = analyze_ulcer_segmentation(image_path)
                measurements = summarize_ulcer_measurements(ulcer)
                overlay_path = ulcer.get("overlay_path")
                if overlay_path:
                    image_items.append({"path": overlay_path, "caption": "FUSeg ulcer-like segmentation overlay"})

                foot_roi_path = None
                if foot_roi_mask_file is not None:
                    foot_roi_path = save_uploaded_file(foot_roi_mask_file, "foot_roi_mask")
                    image_items.append({"path": str(foot_roi_path), "caption": "User-supplied binary visible-foot ROI mask"})

                advanced = calculate_advanced_wound_measurements(
                    segmentation_mask=ulcer.get("mask"),
                    original_image_path=image_path,
                    foot_roi_mask_path=foot_roi_path,
                    reference_length_cm=reference_length_cm,
                    reference_length_pixels=reference_length_pixels,
                )

                segmentation_lines = [
                    ulcer.get("summary"),
                    f"Detected regions: {ulcer.get('number_of_regions')}",
                    f"Model-space predicted area: {ulcer.get('total_area_pixels')} pixels on the 256×256 inference mask",
                    f"Original-resolution mapped area: {advanced['wound_area_pixels_original']} pixels",
                    f"Mean predicted probability inside the segmented mask: {float(ulcer.get('confidence') or 0):.1%}",
                    "The probability above is not diagnostic confidence and does not measure severity.",
                    f"Area percentage of original image: {advanced['area_percent_of_original_image']}%",
                    f"Dominant image zone: {measurements.get('dominant_ulcer_zone')}",
                ]
                if advanced.get("foot_relative_area_percent") is not None:
                    segmentation_lines.append(
                        f"Wound-to-visible-foot area ratio: {advanced['foot_relative_area_percent']}% using the supplied foot ROI mask"
                    )
                else:
                    segmentation_lines.append("Wound-to-visible-foot area ratio: Not available")
                if advanced.get("estimated_area_cm2") is not None:
                    segmentation_lines.append(
                        f"Approximate calibrated planar area: {advanced['estimated_area_cm2']} cm²"
                    )
                else:
                    segmentation_lines.append("Approximate physical area: Not available")
                segmentation_lines.extend(advanced.get("measurement_notes", []))
                segmentation_lines.append(ulcer.get("safety_note"))

                sections.append(
                    {
                        "heading": "Visible ulcer-like segmentation and measurements",
                        "lines": segmentation_lines,
                    }
                )

                pathway = analyze_complication_pathways(
                    rgb_result={
                        "area_pixels": ulcer.get("total_area_pixels", 0),
                        "predicted_area_pixels": ulcer.get("total_area_pixels", 0),
                    },
                    clinical_inputs=inputs,
                    thermal_result=None,
                    previous_area_pixels=(previous_area_pixels or None),
                )
                sections.append(
                    {
                        "heading": "Rule-based complication review pathways",
                        "lines": [
                            f"Primary pathway: {pathway['primary_pathway']}",
                            f"Escalation priority: {pathway['escalation_priority']}",
                            f"Infection review flag: {pathway['infection_review_flag']}",
                            f"Vascular review flag: {pathway['vascular_review_flag']}",
                            f"Delayed-healing flag: {pathway['delayed_healing_flag']}",
                            f"Bone-involvement review flag: {pathway['bone_involvement_review_flag']}",
                            f"Assessed clinical fields: {pathway.get('assessed_field_count')}",
                            f"Unassessed clinical fields: {pathway.get('unassessed_field_count')}",
                        ]
                        + pathway.get("reasons", [])
                        + [pathway.get("safety_note")],
                    }
                )

        elif workflow == WORKFLOW_STANDUP:'''
source, count = ulcer_block_pattern.subn(ulcer_block, source, count=1)
if count != 1:
    raise RuntimeError("Could not install advanced wound measurement output.")

# Expose only workflows with verified provenance.
source = source.replace(
    '''workflow = st.radio(
    "Select analysis workflow",
    [WORKFLOW_ULCER, WORKFLOW_STANDUP, WORKFLOW_PSEUDOCOLOR],
    horizontal=True,
)''',
    '''st.info(
    "Verified professional workflows: close-up RGB ulcer-like segmentation and matched "
    "STANDUP plantar RGB + grayscale thermal fusion."
)
workflow = st.radio(
    "Select analysis workflow",
    [WORKFLOW_ULCER, WORKFLOW_STANDUP],
    horizontal=True,
)''',
    1,
)
source = source.replace(
    "Unified, workflow-safe diabetic-foot screening-support and documentation platform",
    "Verified RGB segmentation and paired RGB–grayscale thermal research platform",
    1,
)

namespace = {"__file__": str(BASE_APP), "__name__": "qadamcare_professional_v3_runtime"}
exec(compile(source, str(BASE_APP), "exec"), namespace)
