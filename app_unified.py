import sys
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from input_modality_router import (
    inspect_image,
    validate_pseudocolor_thermal,
    validate_standup_pair,
    validate_wound_rgb,
)
from llm_report_polisher import polish_report_with_llm
from measurement_support import (
    compute_image_quality_score,
    summarize_thermal_measurements,
    summarize_ulcer_measurements,
)
from multimodal_clinical_copilot import build_multimodal_summary
from secondary_complication_engine import analyze_complication_pathways
from standup_inference import analyze_dm_control_pattern
from thermal_inference import analyze_thermal_with_model, save_attention_images
from thermal_risk_zones import analyze_thermal_risk_zones, save_thermal_risk_outputs
from ulcer_segmentation_inference import analyze_ulcer_segmentation
from unified_pdf_report import generate_unified_pdf_report


APP_OUTPUTS = ROOT / "outputs" / "unified_app"
APP_OUTPUTS.mkdir(parents=True, exist_ok=True)

SAFETY_TEXT = (
    "QadamCare AI is an educational engineering prototype for screening-support, "
    "monitoring documentation, and research demonstration. It does not diagnose diabetes, "
    "diabetic-foot ulcer, infection, ischemia, osteomyelitis, wound depth, severity, future "
    "ulcer location, or treatment need. Every result requires qualified clinician review."
)

WORKFLOW_ULCER = "Visible ulcer analysis"
WORKFLOW_STANDUP = "STANDUP paired RGB + grayscale thermal"
WORKFLOW_PSEUDOCOLOR = "Pseudo-colour thermal research analysis"


st.set_page_config(
    page_title="QadamCare AI Unified",
    page_icon="🦶",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {max-width: 1500px; padding-top: 1.7rem; padding-bottom: 3rem;}
.qc-title {font-size: 3rem; font-weight: 850; letter-spacing: -1px; margin-bottom: .2rem;}
.qc-subtitle {font-size: 1.15rem; color: #9aa4b2; margin-bottom: 1.1rem;}
.qc-card {border: 1px solid rgba(255,255,255,.12); border-radius: 16px; padding: 1rem 1.1rem; margin: .5rem 0 1rem 0; background: rgba(255,255,255,.045);}
.qc-info {border-left: 5px solid #3b82f6; background: rgba(59,130,246,.11); border-radius: 12px; padding: .9rem 1rem; margin: .7rem 0;}
.qc-warning {border-left: 5px solid #f59e0b; background: rgba(245,158,11,.12); border-radius: 12px; padding: .9rem 1rem; margin: .7rem 0;}
.qc-danger {border-left: 5px solid #ef4444; background: rgba(239,68,68,.12); border-radius: 12px; padding: .9rem 1rem; margin: .7rem 0;}
.qc-success {border-left: 5px solid #16a34a; background: rgba(22,163,74,.12); border-radius: 12px; padding: .9rem 1rem; margin: .7rem 0;}
[data-testid="stMetricValue"] {font-size: 1.85rem;}
</style>
""",
    unsafe_allow_html=True,
)


def save_uploaded_file(uploaded_file, prefix):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        suffix = ".png"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = APP_OUTPUTS / f"{prefix}_{stamp}{suffix}"
    path.write_bytes(uploaded_file.getvalue())
    return path


def image_quality_from_path(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "status": "FAIL",
            "metrics": {},
            "warnings": ["Image could not be read."],
        }
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    warnings = []
    if min(h, w) < 256:
        warnings.append("Low resolution")
    if blur < 25:
        warnings.append("Possible blur")
    if not 35 <= brightness <= 225:
        warnings.append("Brightness outside preferred range")
    if contrast < 12:
        warnings.append("Low contrast")
    return {
        "status": "PASS" if not warnings else "RETAKE IMAGE",
        "metrics": {
            "width_px": int(w),
            "height_px": int(h),
            "blur_score": round(blur, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
        },
        "warnings": warnings,
    }


def build_clinical_inputs():
    return {
        "pain_level": pain_level,
        "redness": redness,
        "swelling": swelling,
        "warmth": warmth,
        "discharge": discharge,
        "fever": fever,
        "neuropathy": neuropathy,
        "vascular_disease": vascular_disease,
        "probe_to_bone": probe_to_bone,
    }


def clinical_lines(inputs):
    return [
        f"Pain level entered: {inputs['pain_level']}/10",
        f"Redness reported: {'Yes' if inputs['redness'] else 'No'}",
        f"Swelling reported: {'Yes' if inputs['swelling'] else 'No'}",
        f"Warmth reported: {'Yes' if inputs['warmth'] else 'No'}",
        f"Discharge or odour reported: {'Yes' if inputs['discharge'] else 'No'}",
        f"Fever/systemic symptoms reported: {'Yes' if inputs['fever'] else 'No'}",
        f"Neuropathy reported: {'Yes' if inputs['neuropathy'] else 'No'}",
        f"Vascular disease reported: {'Yes' if inputs['vascular_disease'] else 'No'}",
        f"Probe-to-bone finding reported: {'Yes' if inputs['probe_to_bone'] else 'No'}",
    ]


def build_markdown_report(data):
    patient = data["patient"]
    lines = [
        "# QadamCare AI Unified Screening-Support Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Workflow:** {data['workflow']}",
        f"**Execution status:** {data['status']}",
        "",
        "## Patient / Visit",
        f"- Patient ID: {patient['id']}",
        f"- Patient name: {patient['name']}",
        f"- Age: {patient['age']}",
        f"- Gender: {patient['gender']}",
        f"- Diabetes type: {patient['diabetes_type']}",
        f"- Visit ID: {patient['visit_id']}",
    ]
    for section in data.get("sections", []):
        lines.extend(["", f"## {section['heading']}"])
        for item in section.get("lines", []):
            lines.append(f"- {item}")
    lines.extend(["", "## Safety limitation", SAFETY_TEXT])
    return "\n".join(lines)


def show_validation(validation, title):
    st.markdown(f"### {title}")
    st.write(validation.get("message", ""))
    if validation.get("warnings"):
        for warning in validation["warnings"]:
            st.warning(warning)
    with st.expander("Format measurements"):
        st.json(validation.get("metrics", {}))


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
    st.header("Follow-Up")
    previous_area_pixels = st.number_input(
        "Previous comparable wound-like area (pixels)",
        min_value=0,
        value=0,
        step=100,
    )

    st.divider()
    st.caption("Clinical findings are user-entered. They are not inferred by the image models.")


st.markdown('<div class="qc-title">QadamCare AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="qc-subtitle">Unified, workflow-safe diabetic-foot screening-support and documentation platform</div>',
    unsafe_allow_html=True,
)
st.markdown(f'<div class="qc-info"><b>Safety:</b> {SAFETY_TEXT}</div>', unsafe_allow_html=True)

workflow = st.radio(
    "Select analysis workflow",
    [WORKFLOW_ULCER, WORKFLOW_STANDUP, WORKFLOW_PSEUDOCOLOR],
    horizontal=True,
)

st.session_state.setdefault("unified_result", None)
st.session_state.setdefault("llm_result", None)
st.session_state.setdefault("vlm_result", None)

uploaded = {}

if workflow == WORKFLOW_ULCER:
    st.markdown("### Required input")
    st.info("Upload one close-up normal RGB image of the foot/wound. Do not upload a full-person photograph or thermal image.")
    uploaded["wound_rgb"] = st.file_uploader(
        "Close-up RGB foot/wound image",
        type=["png", "jpg", "jpeg"],
        key="ulcer_rgb",
    )
elif workflow == WORKFLOW_STANDUP:
    st.markdown("### Required paired inputs")
    st.info("Upload a matching plantar RGB image and STANDUP-style grayscale thermal image from the same capture/participant.")
    c1, c2 = st.columns(2)
    with c1:
        uploaded["standup_rgb"] = st.file_uploader(
            "STANDUP plantar RGB image",
            type=["png", "jpg", "jpeg"],
            key="standup_rgb",
        )
    with c2:
        uploaded["standup_thermal"] = st.file_uploader(
            "STANDUP grayscale thermal image",
            type=["png", "jpg", "jpeg"],
            key="standup_thermal",
        )
else:
    st.markdown("### Required input")
    st.info("Upload one pseudo-coloured plantar thermogram, such as a red/yellow/green thermal image.")
    uploaded["pseudo_thermal"] = st.file_uploader(
        "Pseudo-colour plantar thermogram",
        type=["png", "jpg", "jpeg"],
        key="pseudo_thermal",
    )


if st.button("Run validated analysis", type="primary", use_container_width=True):
    st.session_state["llm_result"] = None
    st.session_state["vlm_result"] = None
    patient = {
        "id": patient_id,
        "name": patient_name,
        "age": age,
        "gender": gender,
        "diabetes_type": diabetes_type,
        "visit_id": visit_id,
    }
    inputs = build_clinical_inputs()
    sections = [{"heading": "Clinician-entered findings", "lines": clinical_lines(inputs)}]
    image_items = []
    execution_status = "COMPLETED"
    primary_image_path = None
    overlay_path = None

    try:
        if workflow == WORKFLOW_ULCER:
            file = uploaded.get("wound_rgb")
            if file is None:
                raise ValueError("Upload a close-up RGB foot/wound image first.")
            image_path = save_uploaded_file(file, "wound_rgb")
            primary_image_path = image_path
            validation = validate_wound_rgb(image_path)
            quality = image_quality_from_path(image_path)
            quality_score = compute_image_quality_score(quality)
            sections.append(
                {
                    "heading": "Input validation and quality",
                    "lines": [
                        f"Applicability status: {validation['status']}",
                        validation["message"],
                        f"Image quality status: {quality['status']}",
                        f"Image quality score: {quality_score['image_quality_score']}/100",
                    ]
                    + validation.get("warnings", [])
                    + quality.get("warnings", []),
                }
            )
            image_items.append({"path": str(image_path), "caption": "Submitted close-up RGB image"})

            if not validation["is_valid"] or quality["status"] != "PASS":
                execution_status = "BLOCKED — RETAKE / CORRECT INPUT REQUIRED"
                sections.append(
                    {
                        "heading": "Model execution",
                        "lines": [
                            "FUSeg ulcer segmentation was not run.",
                            "No lesion count, area, review level, complication pathway, or AI overlay was generated from this invalid input.",
                        ],
                    }
                )
            else:
                ulcer = analyze_ulcer_segmentation(image_path)
                measurements = summarize_ulcer_measurements(ulcer)
                overlay_path = ulcer.get("overlay_path")
                if overlay_path:
                    image_items.append({"path": overlay_path, "caption": "FUSeg ulcer-like segmentation overlay"})
                sections.append(
                    {
                        "heading": "Visible ulcer-like segmentation",
                        "lines": [
                            ulcer.get("summary"),
                            f"Detected regions: {ulcer.get('number_of_regions')}",
                            f"Predicted area: {ulcer.get('total_area_pixels')} pixels",
                            f"Segmentation confidence: {float(ulcer.get('confidence') or 0):.1%}",
                            f"Area percentage of analysed image: {measurements.get('area_percent_of_image')}%",
                            f"Dominant image zone: {measurements.get('dominant_ulcer_zone')}",
                            ulcer.get("safety_note"),
                        ],
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
                        ]
                        + pathway.get("reasons", [])
                        + [pathway.get("safety_note")],
                    }
                )

        elif workflow == WORKFLOW_STANDUP:
            rgb_file = uploaded.get("standup_rgb")
            thermal_file = uploaded.get("standup_thermal")
            if rgb_file is None or thermal_file is None:
                raise ValueError("Upload both matching STANDUP RGB and grayscale thermal images.")
            rgb_path = save_uploaded_file(rgb_file, "standup_rgb")
            thermal_path = save_uploaded_file(thermal_file, "standup_thermal")
            primary_image_path = rgb_path
            validation = validate_standup_pair(rgb_path, thermal_path)
            sections.append(
                {
                    "heading": "Paired input validation",
                    "lines": [
                        f"Validation status: {validation['status']}",
                        validation["message"],
                        validation["safety_note"],
                    ]
                    + validation.get("warnings", []),
                }
            )
            image_items.extend(
                [
                    {"path": str(rgb_path), "caption": "Submitted STANDUP plantar RGB image"},
                    {"path": str(thermal_path), "caption": "Submitted STANDUP grayscale thermal image"},
                ]
            )
            if not validation["is_valid"]:
                execution_status = "BLOCKED — MATCHED STANDUP INPUTS REQUIRED"
                sections.append(
                    {
                        "heading": "Model execution",
                        "lines": [
                            "The RGB–thermal fusion model was not run because the paired-input contract failed.",
                            "No diabetic-foot-like/control-like image-pattern probability was generated.",
                        ],
                    }
                )
            else:
                dm = analyze_dm_control_pattern(rgb_path, thermal_path)
                sections.append(
                    {
                        "heading": "STANDUP RGB–thermal image-pattern model",
                        "lines": [
                            f"Model available: {dm.get('model_available')}",
                            f"Predicted pattern: {dm.get('predicted_pattern')}",
                            f"Dataset-defined diabetic-foot-like probability: {dm.get('diabetic_foot_pattern_probability')}",
                            f"Dataset-defined healthy/control-like probability: {dm.get('healthy_control_pattern_probability')}",
                            dm.get("summary"),
                            dm.get("safety_note"),
                        ],
                    }
                )
                zones = analyze_thermal_risk_zones(thermal_path)
                zone_paths = save_thermal_risk_outputs(zones, APP_OUTPUTS)
                overlay_path = zone_paths.get("overlay_path")
                image_items.append({"path": overlay_path, "caption": "Relative thermal monitoring-zone overlay"})
                thermal_measurements = summarize_thermal_measurements(zones)
                sections.append(
                    {
                        "heading": "Relative thermal monitoring-zone measurements",
                        "lines": [
                            f"Overall monitoring zone: {zones['overall_monitoring_zone']}",
                            f"High-monitoring ratio: {thermal_measurements['hotspot_ratio_percent']}%",
                            f"Medium-monitoring ratio: {thermal_measurements['medium_zone_ratio_percent']}%",
                            f"Left-right asymmetry score: {thermal_measurements['asymmetry_score']}",
                            f"Dominant high-monitoring image zone: {thermal_measurements['dominant_high_monitoring_zone']}",
                            zones["safety_note"],
                        ],
                    }
                )

        else:
            file = uploaded.get("pseudo_thermal")
            if file is None:
                raise ValueError("Upload a pseudo-coloured plantar thermogram first.")
            thermal_path = save_uploaded_file(file, "pseudo_thermal")
            primary_image_path = thermal_path
            validation = validate_pseudocolor_thermal(thermal_path)
            sections.append(
                {
                    "heading": "Pseudo-colour thermal validation",
                    "lines": [
                        f"Validation status: {validation['status']}",
                        validation["message"],
                        validation["safety_note"],
                    ]
                    + validation.get("warnings", []),
                }
            )
            image_items.append({"path": str(thermal_path), "caption": "Submitted pseudo-colour thermogram"})
            if not validation["is_valid"]:
                execution_status = "BLOCKED — PSEUDO-COLOUR THERMOGRAM REQUIRED"
                sections.append(
                    {
                        "heading": "Model execution",
                        "lines": [
                            "The legacy pseudo-colour thermal-only classifier was not run.",
                            "The STANDUP paired model was also not run because it requires a grayscale thermal image plus matching RGB image.",
                        ],
                    }
                )
            else:
                thermal = analyze_thermal_with_model(thermal_path)
                attention_paths = save_attention_images(thermal, APP_OUTPUTS)
                overlay_path = attention_paths.get("attention_overlay_path")
                image_items.extend(
                    [
                        {"path": attention_paths.get("attention_map_path"), "caption": "Thermal model attention map"},
                        {"path": overlay_path, "caption": "Thermal model attention overlay"},
                    ]
                )
                sections.append(
                    {
                        "heading": "Pseudo-colour thermal-only research model",
                        "lines": [
                            f"Predicted pattern: {thermal.get('predicted_pattern')}",
                            f"Dataset-defined DM-group pattern probability: {thermal.get('dm_probability')}",
                            f"Decision threshold: {thermal.get('threshold')}",
                            thermal.get("summary"),
                            thermal.get("safety_note"),
                        ],
                    }
                )
                zones = analyze_thermal_risk_zones(thermal_path)
                zone_paths = save_thermal_risk_outputs(zones, APP_OUTPUTS)
                image_items.append({"path": zone_paths.get("overlay_path"), "caption": "Relative intensity monitoring-zone overlay"})
                measurements = summarize_thermal_measurements(zones)
                sections.append(
                    {
                        "heading": "Relative intensity measurements",
                        "lines": [
                            f"Overall monitoring zone: {zones['overall_monitoring_zone']}",
                            f"High-monitoring ratio: {measurements['hotspot_ratio_percent']}%",
                            f"Medium-monitoring ratio: {measurements['medium_zone_ratio_percent']}%",
                            f"Asymmetry score: {measurements['asymmetry_score']}",
                            "These are display-intensity measurements, not calibrated temperatures.",
                        ],
                    }
                )

        data = {
            "workflow": workflow,
            "status": execution_status,
            "patient": patient,
            "clinical_inputs": inputs,
            "sections": sections,
            "images": image_items,
            "primary_image_path": str(primary_image_path) if primary_image_path else None,
            "overlay_path": str(overlay_path) if overlay_path else None,
        }
        data["markdown"] = build_markdown_report(data)
        st.session_state["unified_result"] = data
    except Exception as error:
        st.session_state["unified_result"] = {
            "workflow": workflow,
            "status": "ERROR / INPUT REQUIRED",
            "patient": patient,
            "clinical_inputs": inputs,
            "sections": [
                {
                    "heading": "Execution error",
                    "lines": [str(error), "No clinical or model conclusion should be drawn from this failed run."],
                }
            ],
            "images": [],
            "primary_image_path": None,
            "overlay_path": None,
        }
        st.session_state["unified_result"]["markdown"] = build_markdown_report(st.session_state["unified_result"])


result = st.session_state.get("unified_result")
if result:
    st.divider()
    st.markdown("## Validated Analysis Summary")
    m1, m2 = st.columns(2)
    m1.metric("Workflow", result["workflow"])
    m2.metric("Execution status", result["status"])

    if result["status"] == "COMPLETED":
        st.markdown('<div class="qc-success"><b>Completed:</b> only workflow-compatible modules were executed.</div>', unsafe_allow_html=True)
    elif result["status"].startswith("BLOCKED"):
        st.markdown('<div class="qc-warning"><b>Blocked safely:</b> incompatible or low-quality input was not sent to a model.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="qc-danger"><b>Not completed:</b> review the execution message.</div>', unsafe_allow_html=True)

    for section in result.get("sections", []):
        st.markdown(f"### {section['heading']}")
        for line in section.get("lines", []):
            if line:
                st.write("-", line)

    valid_images = [item for item in result.get("images", []) if item.get("path") and Path(item["path"]).exists()]
    if valid_images:
        st.markdown("### Images")
        columns = st.columns(min(3, len(valid_images)))
        for index, item in enumerate(valid_images):
            columns[index % len(columns)].image(item["path"], caption=item.get("caption"), use_container_width=True)

    st.markdown("### Structured report")
    st.download_button(
        "Download Markdown report",
        data=result["markdown"],
        file_name=f"qadamcare_{visit_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    pdf_path = APP_OUTPUTS / f"qadamcare_unified_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pdf"
    try:
        generate_unified_pdf_report(
            output_path=pdf_path,
            title="QadamCare AI Unified Screening-Support Report",
            workflow=result["workflow"],
            status=result["status"],
            patient=result["patient"],
            sections=result["sections"],
            images=result.get("images"),
            safety_text=SAFETY_TEXT,
        )
        st.download_button(
            "Download PDF report",
            data=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as error:
        st.warning(f"PDF report could not be generated: {error}")

    st.markdown("### Local AI documentation tools")
    ai1, ai2 = st.columns(2)
    with ai1:
        if st.button("Polish report with local LLM", use_container_width=True):
            st.session_state["llm_result"] = polish_report_with_llm(result["markdown"])
    with ai2:
        vlm_disabled = not result.get("primary_image_path") or not Path(result["primary_image_path"]).exists()
        if st.button("Generate local VLM documentation note", disabled=vlm_disabled, use_container_width=True):
            st.session_state["vlm_result"] = build_multimodal_summary(
                structured_report_markdown=result["markdown"],
                rgb_image_path=result.get("primary_image_path"),
                rgb_overlay_path=result.get("overlay_path"),
            )

    llm_result = st.session_state.get("llm_result")
    if llm_result:
        if llm_result.get("available"):
            st.markdown("#### Local LLM-polished report")
            st.markdown(llm_result["text"])
        else:
            st.info(llm_result.get("message"))

    vlm_result = st.session_state.get("vlm_result")
    if vlm_result:
        if vlm_result.get("available"):
            st.markdown("#### Local VLM documentation note")
            st.markdown(vlm_result["text"])
        else:
            st.info(vlm_result.get("message"))

    st.caption(SAFETY_TEXT)
