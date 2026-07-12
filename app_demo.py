
import sys
from pathlib import Path
from datetime import datetime

import cv2
import torch
import pandas as pd
import streamlit as st
import albumentations as A


# ============================================================
# 1. Project paths and internal imports
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import build_model
from quality import assess_image_quality
from features import extract_lesion_features
from tracker import compare_visits
from risk import estimate_risk_level
from pdf_report import generate_pdf_report
from clinical_ai import generate_clinical_support_summary
from clinical_inputs import summarize_clinical_inputs
from advanced_clinical import advanced_clinical_analysis
from feature_status import FEATURE_STATUS
from multimodal_plan import get_multimodal_summary
from fusion_engine import compute_fusion_decision
from thermal_inference import analyze_thermal_with_model
from thermal_quality import validate_thermal_image
from clinical_report_engine import build_clinical_report, report_to_markdown
from llm_report_polisher import polish_report_with_llm
from multimodal_clinical_copilot import build_multimodal_summary
from secondary_complication_engine import analyze_complication_pathways


MODEL_PATH = ROOT / "outputs" / "models" / "unet_efficientnet_b0_25epochs_best.pth"
APP_OUTPUTS = ROOT / "outputs" / "app_reports"
APP_OUTPUTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. UI styling
# ============================================================

st.set_page_config(
    page_title="QadamCare AI",
    page_icon="🦶",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --qc-bg-card: rgba(255, 255, 255, 0.055);
    --qc-bg-card-strong: rgba(255, 255, 255, 0.085);
    --qc-border: rgba(255, 255, 255, 0.12);
    --qc-muted: #9aa4b2;
    --qc-blue: #3b82f6;
    --qc-green: #16a34a;
    --qc-orange: #f59e0b;
    --qc-red: #ef4444;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

.main-title {
    font-size: 48px;
    font-weight: 850;
    letter-spacing: -1.2px;
    margin-bottom: 0.25rem;
}

.subtitle {
    font-size: 20px;
    color: var(--qc-muted);
    margin-bottom: 1.2rem;
}

.section-title {
    font-size: 1.55rem;
    font-weight: 760;
    margin-top: 1.35rem;
    margin-bottom: 0.35rem;
}

.section-note {
    color: var(--qc-muted);
    font-size: 0.96rem;
    margin-bottom: 0.85rem;
}

.qc-card {
    border: 1px solid var(--qc-border);
    background: var(--qc-bg-card);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin: 0.35rem 0 0.9rem 0;
}

.qc-alert {
    border-left: 5px solid var(--qc-blue);
    background: rgba(59, 130, 246, 0.11);
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin: 0.8rem 0 1rem 0;
}

.qc-warning {
    border-left: 5px solid var(--qc-orange);
    background: rgba(245, 158, 11, 0.13);
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin: 0.8rem 0 1rem 0;
}

.qc-critical {
    border-left: 5px solid var(--qc-red);
    background: rgba(239, 68, 68, 0.13);
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin: 0.8rem 0 1rem 0;
}

.qc-small {
    font-size: 0.92rem;
    color: var(--qc-muted);
}

.badge-low, .badge-minimal {
    display: inline-block;
    background-color: #14532d;
    padding: 9px 14px;
    border-radius: 999px;
    color: white;
    font-weight: 700;
}

.badge-moderate {
    display: inline-block;
    background-color: #92400e;
    padding: 9px 14px;
    border-radius: 999px;
    color: white;
    font-weight: 700;
}

.badge-high {
    display: inline-block;
    background-color: #7f1d1d;
    padding: 9px 14px;
    border-radius: 999px;
    color: white;
    font-weight: 700;
}

hr {
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. General helper functions
# ============================================================

def safe_text(value, default="Not provided"):
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def section(title, note=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def safety_banner():
    st.markdown(
        """
<div class="qc-alert">
<b>Clinical safety statement.</b>
QadamCare AI is an educational engineering prototype for screening-support and documentation.
It does not establish diagnosis, infection status, ulcer depth, Wagner grade, ischemia, osteomyelitis,
or a treatment plan. Final interpretation requires qualified clinician assessment.
</div>
""",
        unsafe_allow_html=True,
    )


def make_previous_visit_context(use_previous_visit, previous_visit_id, previous_area_pixels, previous_visit_note):
    return {
        "use_previous_visit": use_previous_visit,
        "previous_visit_id": safe_text(previous_visit_id),
        "previous_area_pixels": previous_area_pixels,
        "previous_area_text": (
            "Not provided" if previous_area_pixels is None else f"{previous_area_pixels} pixels"
        ),
        "previous_visit_note": safe_text(previous_visit_note),
    }


def render_recommendation_box(recommendation, level):
    level = str(level).upper()
    css_class = "qc-critical" if level == "HIGH" else "qc-warning" if level == "MODERATE" else "qc-alert"
    st.markdown(
        f"""
<div class="{css_class}">
<b>Clinician-facing recommendation:</b><br>
{recommendation}
</div>
""",
        unsafe_allow_html=True,
    )


def save_uploaded_file(uploaded_file, prefix):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in [".png", ".jpg", ".jpeg"]:
        suffix = ".png"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = APP_OUTPUTS / f"{prefix}_{timestamp}{suffix}"
    output_path.write_bytes(uploaded_file.getvalue())
    return output_path


# ============================================================
# 4. Model loading and image analysis
# ============================================================

@st.cache_resource
def load_model():
    """Load the RGB segmentation model once per Streamlit session."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model("unet", "efficientnet-b0")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, device


def analyze_image(image_path, previous_area=None, image_size=256):
    """
    Run RGB visible wound/ulcer-like segmentation.

    Output notes:
    - area is in pixels unless calibration is provided elsewhere
    - this function does not diagnose disease or infection
    """
    model, device = load_model()
    image_path = Path(image_path)

    quality = assess_image_quality(image_path)
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"Unable to read image file: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = A.Resize(image_size, image_size)(image=image_rgb)["image"]

    x = (
        torch.tensor(resized, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    ).to(device)

    start_time = datetime.now()

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    inference_time = (datetime.now() - start_time).total_seconds()

    pred_mask = (prob > 0.5).astype("uint8")
    features = extract_lesion_features(pred_mask, prob_map=prob)

    overlay = resized.copy()
    overlay[pred_mask == 1] = [255, 0, 0]
    overlay = cv2.addWeighted(resized, 0.65, overlay, 0.35, 0)

    confidence = float(prob[pred_mask == 1].mean()) if pred_mask.sum() > 0 else 0.0

    risk = estimate_risk_level(
        total_area_pixels=features["total_area_pixels"],
        number_of_lesions=features["number_of_lesions"],
        confidence=confidence,
    )

    visit = None
    if previous_area is not None and previous_area > 0:
        visit = compare_visits(previous_area, features["total_area_pixels"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_save_path = APP_OUTPUTS / f"latest_original_{timestamp}.png"
    overlay_save_path = APP_OUTPUTS / f"latest_overlay_{timestamp}.png"

    cv2.imwrite(str(original_save_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "image": resized,
        "mask": pred_mask,
        "overlay": overlay,
        "quality": quality,
        "features": features,
        "confidence": confidence,
        "visit": visit,
        "risk": risk,
        "inference_time": inference_time,
        "original_path": original_save_path,
        "overlay_path": overlay_save_path,
        "device": str(device),
        "area_pixels": features["total_area_pixels"],
        "predicted_area_pixels": features["total_area_pixels"],
    }


# ============================================================
# 5. Tables and report text
# ============================================================

def lesion_table(features):
    rows = []

    for lesion in features.get("lesions", []):
        rows.append(
            {
                "Region": f"Region {lesion['lesion_id']}",
                "Estimated Area": f"{lesion['area_pixels']} px",
                "Mean Confidence": (
                    f"{lesion['mean_confidence'] * 100:.1f}%"
                    if lesion["mean_confidence"] is not None
                    else "N/A"
                ),
                "Interpretation": "Visible wound/ulcer-like region for clinician review",
            }
        )

    return pd.DataFrame(rows)


def quality_clinical_table(quality):
    metrics = quality["metrics"]

    blur_status = "Acceptable" if metrics["blur_score"] >= 25 else "Retake recommended"
    lighting_status = (
        "Acceptable" if 35 <= metrics["brightness"] <= 225 else "Retake recommended"
    )
    contrast_status = "Acceptable" if metrics["contrast"] >= 12 else "Low contrast"

    return pd.DataFrame(
        [
            {
                "Check": "Image resolution",
                "Result": (
                    "Acceptable"
                    if metrics["width_px"] >= 512 and metrics["height_px"] >= 512
                    else "Low resolution"
                ),
            },
            {"Check": "Focus / blur", "Result": blur_status},
            {"Check": "Lighting", "Result": lighting_status},
            {"Check": "Contrast", "Result": contrast_status},
            {"Check": "Overall suitability", "Result": quality["status"]},
        ]
    )


def technical_table(quality):
    return pd.DataFrame(
        [
            {"Metric": key.replace("_", " ").title(), "Value": value}
            for key, value in quality["metrics"].items()
        ]
    )


def append_secondary_pathway_to_markdown(
    clinical_report_markdown,
    secondary_complication_result,
    previous_visit_context=None,
):
    if previous_visit_context is not None:
        clinical_report_markdown += "\n\n## Previous Visit / Follow-Up Context\n"
        clinical_report_markdown += (
            f"\n**Previous visit ID:** {previous_visit_context.get('previous_visit_id', 'Not provided')}\n"
        )
        clinical_report_markdown += (
            f"\n**Previous area:** {previous_visit_context.get('previous_area_text', 'Not provided')}\n"
        )
        clinical_report_markdown += (
            f"\n**Previous note:** {previous_visit_context.get('previous_visit_note', 'Not provided')}\n"
        )
        clinical_report_markdown += (
            "\n**Interpretation caution:** Area-change interpretation is meaningful only when previous and current images were captured using similar conditions.\n"
        )

    clinical_report_markdown += "\n\n## Complication Pathway Prediction Support\n"
    clinical_report_markdown += (
        f"\n**Primary pathway:** {secondary_complication_result['primary_pathway']}\n"
    )
    clinical_report_markdown += (
        f"\n**Escalation priority:** {secondary_complication_result['escalation_priority']}\n"
    )
    clinical_report_markdown += (
        f"\n**Infection review flag:** {secondary_complication_result['infection_review_flag']}\n"
    )
    clinical_report_markdown += (
        f"\n**Vascular review flag:** {secondary_complication_result['vascular_review_flag']}\n"
    )
    clinical_report_markdown += (
        f"\n**Delayed-healing flag:** {secondary_complication_result['delayed_healing_flag']}\n"
    )
    clinical_report_markdown += (
        f"\n**Bone-involvement review flag:** {secondary_complication_result['bone_involvement_review_flag']}\n"
    )

    area_change = secondary_complication_result.get("area_change_percent")
    clinical_report_markdown += (
        f"\n**Area change from previous visit:** {area_change:.1f}%\n"
        if area_change is not None
        else "\n**Area change from previous visit:** Not available\n"
    )

    clinical_report_markdown += "\n\n**Reasons:**\n"
    for reason in secondary_complication_result["reasons"]:
        clinical_report_markdown += f"- {reason}\n"

    clinical_report_markdown += (
        f"\n**Safety note:** {secondary_complication_result['safety_note']}\n"
    )

    return clinical_report_markdown


def build_text_report(
    patient,
    result,
    recommendation,
    secondary_complication_result=None,
    previous_visit_context=None,
):
    q = result["quality"]
    f = result["features"]
    r = result["risk"]
    clinical_ai = result["clinical_ai"]
    v = result["visit"]

    text = f"""
QadamCare AI - Clinician Screening-Support Report
=================================================

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Patient / Visit
---------------
Patient ID: {patient["id"]}
Patient Name: {patient["name"]}
Age: {patient["age"]}
Gender: {patient["gender"]}
Diabetes Type: {patient["diabetes_type"]}
Visit ID: {patient["visit_id"]}

Image Quality
-------------
Overall Status: {q["status"]}

AI Segmentation Summary
-----------------------
Detected visible wound/ulcer-like regions: {f["number_of_lesions"]}
Total predicted wound-like area: {f["total_area_pixels"]} pixels
Average confidence: {result["confidence"] * 100:.1f}%

Prototype Review Priority
-------------------------
Review Level: {r["risk_level"]}
Review Score: {r["risk_score"]}
Interpretation: {r["message"]}

AI Clinical Support Analysis
----------------------------
Review Priority: {clinical_ai["review_priority"]}
Healing Score: {clinical_ai["healing_score"]}/100
Clinical Impression: {clinical_ai["clinical_impression"]}
Recommended Action: {clinical_ai["recommended_action"]}
Trend Interpretation: {clinical_ai["trend_note"]}
Follow-up Suggestion: {clinical_ai["follow_up_suggestion"]}
"""

    if previous_visit_context is not None:
        text += f"""

Previous Visit Context
----------------------
Previous Visit ID: {previous_visit_context.get("previous_visit_id", "Not provided")}
Previous Area: {previous_visit_context.get("previous_area_text", "Not provided")}
Previous Note: {previous_visit_context.get("previous_visit_note", "Not provided")}
"""

    if v is not None:
        text += f"""

Visit-to-Visit Monitoring
-------------------------
Trend: {v["status"]}
Area Change: {v["change_pixels"]} pixels
Change Percentage: {v["change_percent"]}%
Interpretation: {v["message"]}
"""

    if secondary_complication_result is not None:
        text += f"""

Complication Pathway Prediction Support
---------------------------------------
Primary Pathway: {secondary_complication_result["primary_pathway"]}
Escalation Priority: {secondary_complication_result["escalation_priority"]}
Infection Review Flag: {secondary_complication_result["infection_review_flag"]}
Vascular Review Flag: {secondary_complication_result["vascular_review_flag"]}
Delayed-Healing Flag: {secondary_complication_result["delayed_healing_flag"]}
Bone-Involvement Review Flag: {secondary_complication_result["bone_involvement_review_flag"]}

Reasons:
- {chr(10).join(secondary_complication_result["reasons"])}

Safety Note:
{secondary_complication_result["safety_note"]}
"""

    text += f"""

Clinician-Oriented Recommendation
---------------------------------
{recommendation}

Safety Statement
----------------
This is an educational engineering prototype. It supports visual screening,
secondary complication review-prioritization, and documentation only. It does not
confirm diagnosis, infection, ischemia, osteomyelitis, ulcer depth, Wagner grade,
or amputation risk. Qualified clinician assessment is required.
"""

    return text


# ============================================================
# 6. Session state
# ============================================================

DEFAULT_STATE = {
    "analysis_data": None,
    "polished_report": None,
    "multimodal_copilot_report": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 7. Sidebar
# ============================================================

with st.sidebar:
    st.header("Patient Information")

    patient_id = st.text_input("Patient ID", value="QA-00125")
    patient_name = st.text_input("Patient Name", value="Demo Patient")
    age = st.number_input("Age", min_value=1, max_value=120, value=63)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    diabetes_type = st.selectbox(
        "Diabetes Type",
        ["Type II", "Type I", "Gestational", "Not specified"],
    )
    visit_id = st.text_input("Visit ID", value="Visit-1")

    st.divider()
    st.subheader("Previous Visit / Follow-Up")

    use_previous_visit = st.checkbox(
        "Add previous visit data",
        value=False,
        help="Use only when previous area came from a comparable image workflow.",
    )

    previous_visit_id = ""
    previous_area_pixels = None
    previous_visit_note = ""

    if use_previous_visit:
        previous_visit_id = st.text_input("Previous Visit ID", value="Visit-0")
        previous_area_input = st.number_input(
            "Previous wound-like area (pixels)",
            min_value=0,
            value=0,
            step=100,
            help="Meaningful only if the previous and current images were captured similarly.",
        )
        previous_visit_note = st.text_area(
            "Previous visit note",
            value="",
            height=80,
            placeholder="Example: image captured 2 weeks ago.",
        )
        previous_area_pixels = None if previous_area_input == 0 else previous_area_input

    st.divider()
    st.subheader("Measurement Calibration")

    pixels_per_cm = st.number_input(
        "Pixels per centimetre",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Use only when a ruler or validated marker is visible. Leave 0 if unavailable.",
    )

    st.divider()
    st.header("Clinical Findings")

    pain_level = st.slider("Pain level", 0, 10, 0)
    redness = st.checkbox("Redness")
    swelling = st.checkbox("Swelling")
    warmth = st.checkbox("Warmth")
    discharge = st.checkbox("Discharge or odor")
    fever = st.checkbox("Fever / systemic symptoms")
    neuropathy = st.checkbox("Known or suspected neuropathy")
    vascular_disease = st.checkbox("Known or suspected vascular disease")
    probe_to_bone = st.checkbox("Positive probe-to-bone finding")

    st.divider()
    st.header("Workflow")
    st.write("1. Upload image")
    st.write("2. Add clinical findings")
    st.write("3. Run AI analysis")
    st.write("4. Review outputs")
    st.write("5. Export report")


# ============================================================
# 8. Header and upload section
# ============================================================

st.markdown('<div class="main-title">QadamCare AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-assisted diabetic foot screening, monitoring, and clinician documentation platform</div>',
    unsafe_allow_html=True,
)
safety_banner()

section("Input Images", "Upload a standard RGB foot image. Thermal upload is optional and treated as a research extension.")

upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    uploaded_file = st.file_uploader("RGB foot image", type=["png", "jpg", "jpeg"])

with upload_col2:
    thermal_file = st.file_uploader(
        "Optional thermography image",
        type=["png", "jpg", "jpeg"],
        help="Research extension only. Not validated for diagnosis.",
    )


# ============================================================
# 9. Run analysis
# ============================================================

if uploaded_file is None:
    st.info("Upload a foot image to begin analysis.")
else:
    if st.button("Analyze Image", type="primary", use_container_width=True):
        st.session_state["polished_report"] = None
        st.session_state["multimodal_copilot_report"] = None

        with st.spinner("Running QadamCare AI analysis..."):
            image_path = save_uploaded_file(uploaded_file, prefix="uploaded_rgb")

            thermal_result = None
            thermal_quality = None
            thermal_path = None

            if thermal_file is not None:
                thermal_path = save_uploaded_file(thermal_file, prefix="uploaded_thermal")
                thermal_quality = validate_thermal_image(thermal_path)

                if thermal_quality["is_valid"]:
                    thermal_result = analyze_thermal_with_model(thermal_path)

                    if isinstance(thermal_result, dict) and "predicted_pattern" in thermal_result:
                        thermal_result["predicted_label"] = thermal_result["predicted_pattern"]

                    thermal_result["thermal_available"] = True
                else:
                    thermal_result = {
                        "thermal_available": False,
                        "thermal_concern": "INVALID / REVIEW NEEDED",
                        "hot_region_ratio": None,
                        "mean_intensity": None,
                        "max_intensity": None,
                        "note": thermal_quality["message"],
                        "reasons": thermal_quality["warnings"],
                    }

            patient = {
                "id": patient_id,
                "name": patient_name,
                "age": age,
                "gender": gender,
                "diabetes_type": diabetes_type,
                "visit_id": visit_id,
            }

            clinical_inputs = {
                "pain_level": pain_level,
                "redness": redness,
                "swelling": swelling,
                "warmth": warmth,
                "discharge": discharge,
                "fever": fever,
                "neuropathy": neuropathy,
                "vascular_disease": vascular_disease,
                "probe_to_bone": probe_to_bone,
                "pixels_per_cm": pixels_per_cm,
            }

            previous_visit_context = make_previous_visit_context(
                use_previous_visit=use_previous_visit,
                previous_visit_id=previous_visit_id,
                previous_area_pixels=previous_area_pixels,
                previous_visit_note=previous_visit_note,
            )

            result = analyze_image(
                image_path=image_path,
                previous_area=previous_area_pixels,
            )

            result["thermal_quality"] = thermal_quality
            result["thermal_result"] = thermal_result

            quality = result["quality"]
            features = result["features"]
            risk = result["risk"]
            visit = result["visit"]
            confidence = result["confidence"]

            clinical_ai = generate_clinical_support_summary(
                quality=quality,
                features=features,
                confidence=confidence,
                risk=risk,
                visit=visit,
            )

            clinical_input_summary = summarize_clinical_inputs(clinical_inputs)
            result["clinical_input_summary"] = clinical_input_summary

            advanced_ai = advanced_clinical_analysis(
                features=features,
                confidence=confidence,
                risk=risk,
                clinical_inputs=clinical_inputs,
                diabetes_type=diabetes_type,
                visit=visit,
                pixels_per_cm=clinical_inputs["pixels_per_cm"],
            )

            fusion_result = compute_fusion_decision(
                rgb_result=result,
                clinical_summary=clinical_input_summary,
                advanced_ai=advanced_ai,
                thermal_result=None,
            )

            secondary_complication_result = analyze_complication_pathways(
                rgb_result=result,
                clinical_inputs=clinical_inputs,
                thermal_result=thermal_result,
                previous_area_pixels=previous_area_pixels,
            )

            if clinical_input_summary["clinical_concern"] == "HIGH":
                clinical_ai["review_priority"] = "PRIORITY REVIEW"
                clinical_ai["urgency"] = "PRIORITY REVIEW"
                clinical_ai["recommended_action"] = (
                    clinical_ai["recommended_action"]
                    + " Clinician-entered findings increase concern and should be reviewed carefully."
                )
                clinical_ai["reasoning_points"].append(
                    "Clinician-entered findings increased the overall concern level."
                )

            result["clinical_ai"] = clinical_ai
            result["fusion_result"] = fusion_result
            result["advanced_ai"] = advanced_ai

            st.session_state["analysis_data"] = {
                "patient": patient,
                "clinical_inputs": clinical_inputs,
                "previous_visit_context": previous_visit_context,
                "image_path": image_path,
                "thermal_path": thermal_path,
                "thermal_quality": thermal_quality,
                "thermal_result": thermal_result,
                "result": result,
                "quality": quality,
                "features": features,
                "risk": risk,
                "visit": visit,
                "confidence": confidence,
                "clinical_ai": clinical_ai,
                "clinical_input_summary": clinical_input_summary,
                "advanced_ai": advanced_ai,
                "fusion_result": fusion_result,
                "secondary_complication_result": secondary_complication_result,
            }

        st.success("Analysis completed.")


# ============================================================
# 10. Render analysis
# ============================================================

analysis_data = st.session_state.get("analysis_data")

if analysis_data is not None:
    patient = analysis_data["patient"]
    clinical_inputs = analysis_data["clinical_inputs"]
    previous_visit_context = analysis_data["previous_visit_context"]
    thermal_path = analysis_data["thermal_path"]
    thermal_quality = analysis_data["thermal_quality"]
    thermal_result = analysis_data["thermal_result"]
    result = analysis_data["result"]
    quality = analysis_data["quality"]
    features = analysis_data["features"]
    risk = analysis_data["risk"]
    visit = analysis_data["visit"]
    confidence = analysis_data["confidence"]
    clinical_ai = analysis_data["clinical_ai"]
    clinical_input_summary = analysis_data["clinical_input_summary"]
    advanced_ai = analysis_data["advanced_ai"]
    fusion_result = analysis_data["fusion_result"]
    secondary_complication_result = analysis_data["secondary_complication_result"]

    patient_id = patient["id"]
    visit_id = patient["visit_id"]
    previous_area_pixels = previous_visit_context["previous_area_pixels"]

    st.divider()

    # ------------------------------------------------------------
    # 10.1 Executive clinical handoff
    # ------------------------------------------------------------

    section(
        "Clinical Handoff Summary",
        "Fast overview for presentation and clinician review. These outputs are screening-support only.",
    )

    if quality["status"] != "PASS":
        recommendation = "Image quality is not sufficient. Retake the image before relying on visual screening output."
    elif risk["risk_level"] == "HIGH":
        recommendation = "Prioritized clinician review is recommended because the prototype detected a large or high-confidence visible wound/ulcer-like region."
    elif features["number_of_lesions"] > 0:
        recommendation = "Clinician review is recommended. The AI output should be used as screening support only."
    else:
        recommendation = "No major visible wound/ulcer-like region was detected by the prototype. Clinical review may still be needed depending on symptoms and risk factors."

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Image Quality", quality["status"])
    h2.metric("Detected Regions", features["number_of_lesions"])
    h3.metric("Review Level", risk["risk_level"])
    h4.metric("Pathway", secondary_complication_result["primary_pathway"])

    render_recommendation_box(recommendation, risk["risk_level"])

    # ------------------------------------------------------------
    # 10.2 Visual AI result
    # ------------------------------------------------------------

    section("Visual AI Results", "Original image, AI-predicted mask, and overlay for clinician review.")

    c1, c2, c3 = st.columns(3)
    c1.image(result["image"], caption="Original RGB image", use_container_width=True)
    c2.image(result["mask"] * 255, caption="AI predicted wound/ulcer-like mask", use_container_width=True)
    c3.image(result["overlay"], caption="AI overlay for clinician review", use_container_width=True)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Predicted Area", f"{features['total_area_pixels']} px")
    s2.metric("Confidence", f"{confidence * 100:.1f}%")
    s3.metric("Inference Time", f"{result['inference_time']:.4f} s")
    s4.metric("Device", result["device"])

    lesion_df = lesion_table(features)
    with st.expander("Region-level measurements"):
        if lesion_df.empty:
            st.write("No region-level measurements available.")
        else:
            st.dataframe(lesion_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------
    # 10.3 Complication pathway and previous visit
    # ------------------------------------------------------------

    section(
        "Complication Pathway Prediction Support",
        "Rule-based secondary complication review pathway. It prioritizes review; it does not diagnose.",
    )

    path_col1, path_col2, path_col3 = st.columns(3)

    with path_col1:
        st.metric("Primary Pathway", secondary_complication_result["primary_pathway"])

    with path_col2:
        st.metric("Escalation Priority", secondary_complication_result["escalation_priority"])

    with path_col3:
        area_change = secondary_complication_result.get("area_change_percent")
        if area_change is None:
            st.metric("Area Change", "No previous visit")
        else:
            if area_change < -10:
                change_status = "Possible improvement"
            elif area_change > 10:
                change_status = "Possible worsening"
            else:
                change_status = "Stable range"
            st.metric("Area Change", f"{area_change:.1f}%", delta=change_status)

    flag_col1, flag_col2, flag_col3, flag_col4 = st.columns(4)
    flag_col1.metric("Infection Review", secondary_complication_result["infection_review_flag"])
    flag_col2.metric("Vascular Review", secondary_complication_result["vascular_review_flag"])
    flag_col3.metric("Delayed Healing", secondary_complication_result["delayed_healing_flag"])
    flag_col4.metric("Bone Review", secondary_complication_result["bone_involvement_review_flag"])

    if previous_area_pixels is None:
        st.warning("No previous visit data was entered. Follow-up trend interpretation is limited.")
    else:
        st.success(
            f"Previous visit area entered: {previous_area_pixels} pixels. "
            "Area-change interpretation assumes similar image capture conditions."
        )

    with st.expander("Why this pathway was selected"):
        for reason in secondary_complication_result["reasons"]:
            st.write(f"- {reason}")
        st.info(secondary_complication_result["safety_note"])

    section("Previous Visit Context", "Follow-up comparison is meaningful only with similar image capture conditions.")

    pv1, pv2, pv3 = st.columns(3)
    pv1.metric("Previous Visit ID", previous_visit_context["previous_visit_id"])
    pv2.metric("Previous Area", previous_visit_context["previous_area_text"])

    if previous_area_pixels is None:
        pv3.metric("Area Change", "Not available")
    else:
        current_area = features["total_area_pixels"]
        pv3.metric("Area Change", f"{current_area - previous_area_pixels} px")

    if previous_visit_context["previous_visit_note"] != "Not provided":
        st.info(previous_visit_context["previous_visit_note"])

    # ------------------------------------------------------------
    # 10.4 Clinician inputs and advanced support
    # ------------------------------------------------------------

    section("Clinician-Entered Findings", "Findings entered by the user; these are not inferred by the image model.")

    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("Clinical Concern", clinical_input_summary["clinical_concern"])
    ci2.metric("Pain Level", clinical_inputs["pain_level"])
    ci3.metric("AI Role", "Decision Support")

    st.write(clinical_input_summary["note"])

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        st.markdown("**Reported concern points**")
        if clinical_input_summary["concern_points"]:
            for item in clinical_input_summary["concern_points"]:
                st.write("-", item)
        else:
            st.write("No routine concern points were entered.")

    with input_col2:
        st.markdown("**Critical clinical flags**")
        if clinical_input_summary["critical_points"]:
            for item in clinical_input_summary["critical_points"]:
                st.error(item)
        else:
            st.write("No critical clinical flags were entered.")

    section("Advanced Screening-Support Outputs", "Cautious rule-based summaries for clinician review.")

    ad1, ad2, ad3 = st.columns(3)
    ad1.metric("Infection Review Signal", advanced_ai["infection_suspicion"]["level"])

    wagner_text = advanced_ai["wagner_estimate"]["estimated_grade"]
    if "Grade 3" in wagner_text:
        wagner_short = "Grade 3 concern"
    elif "Grade 1" in wagner_text:
        wagner_short = "Grade 1 pattern"
    elif "No visible" in wagner_text:
        wagner_short = "No visible ulcer"
    else:
        wagner_short = "Needs review"

    ad2.metric("Wagner-Style Support", wagner_short)
    ad3.metric(
        "Estimated Area",
        (
            f'{advanced_ai["size_estimation"]["area_cm2"]} cm²'
            if advanced_ai["size_estimation"]["area_cm2"] is not None
            else "Needs calibration"
        ),
    )

    st.caption(advanced_ai["wagner_estimate"]["explanation"])
    st.write("Diabetes context:", advanced_ai["diabetes_note"])
    st.write("AI final summary:", advanced_ai["final_summary"])

    with st.expander("Detailed advanced support reasoning"):
        st.markdown("**Explainability points**")
        for item in advanced_ai["explainability_points"]:
            st.write("-", item)

        st.markdown("**Infection review-signal reasoning**")
        if advanced_ai["infection_suspicion"]["reasons"]:
            for item in advanced_ai["infection_suspicion"]["reasons"]:
                st.write("-", item)
        else:
            st.write("No infection-related clinical flags were entered.")
        st.write(advanced_ai["infection_suspicion"]["note"])

        st.markdown("**Wagner-style caution**")
        st.write(advanced_ai["wagner_estimate"]["caution"])

    # ------------------------------------------------------------
    # 10.5 Prototype evidence and thermal extension
    # ------------------------------------------------------------

    section("Overall Prototype Evidence Summary", "Rule-based summary combining model output and clinician-entered findings.")

    ev1, ev2 = st.columns(2)
    ev1.metric("Review Concern Level", fusion_result["overall_level"])
    ev2.metric("Prototype Signal Score", fusion_result["fusion_score"])

    st.write("Decision support:", fusion_result["decision_support"])
    st.caption(fusion_result["note"])

    with st.expander("Evidence-summary reasoning"):
        for item in fusion_result["reasons"]:
            st.write("-", item)

    section("Thermography Research Extension", "Optional model output; not included in the main fusion score.")

    if thermal_result is None:
        st.info("No thermography image uploaded.")
    elif thermal_quality is not None and not thermal_quality["is_valid"]:
        st.error("Thermal image validation: review needed")
        st.write(thermal_quality["message"])
        with st.expander("Thermal validation details"):
            st.json(thermal_quality["metrics"])
            for item in thermal_quality.get("warnings", []):
                st.write("- " + item)
        st.warning(thermal_quality.get("safety_note", "Thermal validation is not clinical diagnosis."))
    else:
        th1, th2, th3 = st.columns(3)
        th1.metric("DM Group Pattern Probability", f"{thermal_result['dm_probability']:.1%}")
        th2.metric("Decision Threshold", f"{thermal_result['threshold']:.2f}")
        th3.metric("Predicted Pattern", thermal_result["predicted_pattern"])

        st.write(thermal_result["summary"])

        t1, t2 = st.columns(2)
        t1.image(thermal_result["original_image"], caption="Uploaded thermal image", use_container_width=True)
        t2.image(thermal_result["attention_overlay"], caption="Classifier attention overlay", use_container_width=True)
        st.caption("Attention overlay is not a temperature map or disease-location diagnosis.")
        st.warning(thermal_result["safety_note"])

    # ------------------------------------------------------------
    # 10.6 Quality, validation, and system information
    # ------------------------------------------------------------

    section("Image Quality and System Information", "Technical support information for transparent prototype review.")

    q_left, q_right = st.columns(2)

    with q_left:
        st.markdown("**Clinical image-quality checks**")
        st.dataframe(quality_clinical_table(quality), use_container_width=True, hide_index=True)

    with q_right:
        st.markdown("**Technical metrics**")
        st.dataframe(technical_table(quality), use_container_width=True, hide_index=True)

    with st.expander("Implemented feature status"):
        status_rows = [
            {
                "Feature": feature,
                "Status": info["status"],
                "Evidence / Requirement": info["evidence"],
                "Current Use": info["current_use"],
            }
            for feature, info in FEATURE_STATUS.items()
        ]
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    with st.expander("Validation and feasibility roadmap"):
        multimodal_rows = [
            {
                "Module": module_name,
                "Status": info["status"],
                "Purpose": info["purpose"],
                "Validation Requirement": info["validation"],
            }
            for module_name, info in get_multimodal_summary().items()
        ]
        st.dataframe(pd.DataFrame(multimodal_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------
    # 10.7 Structured markdown report and local AI copilot
    # ------------------------------------------------------------

    section("Clinical Decision-Support Summary", "Structured source report used by the PDF export and local AI copilot.")

    clinical_report = build_clinical_report(
        rgb_result=result,
        clinical_inputs=clinical_inputs,
        clinical_summary=clinical_input_summary,
        advanced_ai=advanced_ai,
        thermal_result=thermal_result,
    )

    clinical_report_markdown = report_to_markdown(clinical_report)
    clinical_report_markdown = append_secondary_pathway_to_markdown(
        clinical_report_markdown=clinical_report_markdown,
        secondary_complication_result=secondary_complication_result,
        previous_visit_context=previous_visit_context,
    )

    with st.expander("View structured clinical summary"):
        st.markdown(clinical_report_markdown)

    st.download_button(
        label="Download Clinical Summary (.md)",
        data=clinical_report_markdown,
        file_name=f"qadamcare_clinical_summary_{patient_id}_{visit_id}.md",
        mime="text/markdown",
    )

    section("AI Documentation Copilot", "Local report-polishing and visual documentation assistance. Outputs require clinician review.")

    copilot_col1, copilot_col2 = st.columns(2)

    with copilot_col1:
        if st.button("Polish Structured Report with Local LLM", use_container_width=True):
            with st.spinner("Preparing clinician-facing report language locally..."):
                polished_result = polish_report_with_llm(clinical_report_markdown)

            if polished_result["available"]:
                st.session_state["polished_report"] = polished_result["text"]
                st.success(polished_result["message"])
            else:
                st.warning(polished_result["message"])

        if st.session_state.get("polished_report"):
            st.markdown("### Local LLM-Polished Report")
            st.markdown(st.session_state["polished_report"])
            st.download_button(
                label="Download Polished Report (.md)",
                data=st.session_state["polished_report"],
                file_name="qadamcare_polished_report.md",
                mime="text/markdown",
            )

    with copilot_col2:
        if st.button("Review Images + Report with Local Multimodal Copilot", use_container_width=True):
            with st.spinner("Reviewing structured report and uploaded image locally..."):
                thermal_image_for_copilot = (
                    thermal_path
                    if thermal_result is not None
                    and isinstance(thermal_result, dict)
                    and thermal_result.get("thermal_available")
                    else None
                )

                copilot_result = build_multimodal_summary(
                    structured_report_markdown=clinical_report_markdown,
                    rgb_image_path=result["original_path"],
                    rgb_overlay_path=result["overlay_path"],
                    thermal_image_path=thermal_image_for_copilot,
                    thermal_attention_path=None,
                )

            if copilot_result["available"]:
                st.session_state["multimodal_copilot_report"] = copilot_result["text"]
                st.success(copilot_result["message"])
            else:
                st.warning(copilot_result["message"])

        if st.session_state.get("multimodal_copilot_report"):
            st.markdown("### Doctor-Oriented Multimodal Review")
            st.markdown(st.session_state["multimodal_copilot_report"])
            st.download_button(
                label="Download Copilot Review (.md)",
                data=st.session_state["multimodal_copilot_report"],
                file_name="qadamcare_multimodal_copilot_review.md",
                mime="text/markdown",
            )

    # ------------------------------------------------------------
    # 10.8 PDF and text export
    # ------------------------------------------------------------

    section("Clinician Report Export", "Download the professional PDF report and plain-text summary.")

    pdf_path = APP_OUTPUTS / f"qadamcare_report_{patient_id}_{visit_id}.pdf"

    generate_pdf_report(
        output_path=pdf_path,
        patient_id=patient_id,
        visit_id=visit_id,
        image_path=result["original_path"],
        overlay_path=result["overlay_path"],
        quality=quality,
        features=features,
        confidence=confidence,
        risk=risk,
        recommendation=recommendation,
        visit=visit,
        clinical_ai=clinical_ai,
        clinical_inputs=clinical_inputs,
        clinical_summary=clinical_input_summary,
        advanced_ai=advanced_ai,
        secondary_complication=secondary_complication_result,
        thermal_result=thermal_result,
        fusion_result=fusion_result,
        previous_visit_context={
            "previous_area_pixels": previous_area_pixels,
            "previous_visit_id": previous_visit_context["previous_visit_id"],
            "previous_visit_note": previous_visit_context["previous_visit_note"],
        },
    )

    with open(pdf_path, "rb") as file:
        pdf_bytes = file.read()

    report_text = build_text_report(
        patient=patient,
        result=result,
        recommendation=recommendation,
        secondary_complication_result=secondary_complication_result,
        previous_visit_context=previous_visit_context,
    )

    col_pdf, col_txt = st.columns(2)

    with col_pdf:
        st.download_button(
            label="Download PDF clinician report",
            data=pdf_bytes,
            file_name=f"qadamcare_report_{patient_id}_{visit_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with col_txt:
        st.download_button(
            label="Download text summary",
            data=report_text,
            file_name=f"qadamcare_report_{patient_id}_{visit_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )