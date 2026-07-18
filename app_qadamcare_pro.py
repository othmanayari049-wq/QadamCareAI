from copy import deepcopy
from datetime import datetime
from pathlib import Path

import streamlit as st


# Streamlit reruns the entry script after every interaction. Execute the validated base
# app from source on every rerun, while applying a small safety patch to the patient
# history field. The image model never diagnoses diabetes.
BASE_APP = Path(__file__).resolve().with_name("app_unified.py")

try:
    source = BASE_APP.read_text(encoding="utf-8")

    old_diabetes_widget = '''    diabetes_type = st.selectbox(
        "Diabetes Type",
        ["Type II", "Type I", "Gestational", "Not specified"],
    )'''
    new_diabetes_widget = '''    diabetes_type = st.selectbox(
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
        help=(
            "This is history entered by the user or clinician. It is not inferred by the "
            "RGB, thermal, segmentation, LLM, or VLM models."
        ),
    )
    st.caption("Diabetes history is user-entered. The image model cannot diagnose diabetes.")'''

    if old_diabetes_widget not in source:
        raise RuntimeError(
            "The expected diabetes-history widget was not found in app_unified.py. "
            "Pull the latest branch version before starting the professional app."
        )

    source = source.replace(old_diabetes_widget, new_diabetes_widget, 1)
    source = source.replace(
        'f"- Diabetes type: {patient[\'diabetes_type\']}",',
        'f"- User-entered diabetes history: {patient[\'diabetes_type\']}",',
        1,
    )

    base = {
        "__file__": str(BASE_APP),
        "__name__": "qadamcare_unified_runtime",
    }
    exec(compile(source, str(BASE_APP), "exec"), base)
except Exception as error:
    st.error("QadamCare could not load the validated base application.")
    st.exception(error)
    st.stop()


APP_OUTPUTS = base["APP_OUTPUTS"]
SAFETY_TEXT = base["SAFETY_TEXT"]
polish_report_with_llm = base["polish_report_with_llm"]
build_multimodal_summary = base["build_multimodal_summary"]

from intelligent_pdf_report import generate_intelligent_pdf_report
from local_ollama import OLLAMA_TEXT_MODEL, OLLAMA_VISION_MODEL


def _ai_safe_markdown(result):
    """Add an explicit source-of-truth contract for the local generative models."""
    patient = result.get("patient", {})
    history = patient.get("diabetes_type", "Not specified")
    base_report = result.get("markdown", "")
    distinction = f"""

## Critical interpretation contract
- User-entered diabetes history: {history}
- This history field is not an image-model prediction.
- A STANDUP healthy/control-like image-pattern result does not prove that the person is healthy or does not have diabetes.
- A STANDUP diabetic-foot-like image-pattern result does not diagnose diabetes or diabetic foot.
- Describe diabetes only as user-entered history. Never rewrite the image-pattern result as the person's diabetes status.
- A finding entered as No is explicitly reported absent; it is not missing information.
"""
    return base_report + distinction


def _pdf_safe_result(result):
    """Make the origin of the diabetes field unmistakable inside deterministic PDF tables."""
    safe_result = deepcopy(result)
    patient = safe_result.setdefault("patient", {})
    history = patient.get("diabetes_type", "Not specified")
    patient["diabetes_type"] = f"{history} — user-entered history; not inferred from images"
    safe_result["markdown"] = _ai_safe_markdown(result)
    return safe_result


result = st.session_state.get("unified_result")

if result:
    st.divider()
    st.markdown("## Comprehensive AI-Integrated PDF")
    st.info(
        "This report combines validated analysis, sidebar findings, images and overlays, "
        "evidence-informed LLM reasoning, and the VLM visual note while keeping user-entered "
        "history separate from image-model outputs."
    )

    current_llm = st.session_state.get("llm_result")
    current_vlm = st.session_state.get("vlm_result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Validated analysis", "Included" if result.get("status") else "Unavailable")
    c2.metric(
        "LLM reasoning",
        "Ready" if current_llm and current_llm.get("available") else "Not generated",
    )
    c3.metric(
        "VLM visual note",
        "Ready" if current_vlm and current_vlm.get("available") else "Not generated",
    )

    st.warning(
        "Important: 'healthy/control-like' is an image-pattern result only. It does not decide "
        "whether the person has diabetes. Diabetes history comes only from the sidebar entry."
    )

    include_llm = st.checkbox(
        "Include evidence-informed LLM clinical reasoning",
        value=True,
        help="If it has not already been generated, QadamCare will generate it before building the PDF.",
    )
    include_vlm = st.checkbox(
        "Include VLM visual documentation note",
        value=True,
        help="Requires a valid primary image. If missing, the PDF explains that the VLM section was unavailable.",
    )

    if st.button(
        "Generate comprehensive intelligent PDF",
        type="primary",
        use_container_width=True,
    ):
        with st.status("Building comprehensive report...", expanded=True) as status:
            # Always regenerate AI narratives for the current validated analysis. This avoids
            # carrying an older LLM/VLM narrative into a new patient or model result.
            llm_result = None
            vlm_result = None
            ai_markdown = _ai_safe_markdown(result)

            if include_llm:
                st.write("Generating evidence-informed LLM reasoning...")
                llm_result = polish_report_with_llm(ai_markdown)
                st.session_state["llm_result"] = llm_result
            else:
                llm_result = {
                    "available": False,
                    "message": "LLM reasoning was excluded by the user.",
                    "text": None,
                }

            primary_path = result.get("primary_image_path")
            primary_exists = primary_path and Path(primary_path).exists()
            if include_vlm and primary_exists:
                st.write("Generating VLM visual documentation note...")
                vlm_result = build_multimodal_summary(
                    structured_report_markdown=ai_markdown,
                    rgb_image_path=primary_path,
                    rgb_overlay_path=result.get("overlay_path"),
                )
                st.session_state["vlm_result"] = vlm_result
            elif not include_vlm:
                vlm_result = {
                    "available": False,
                    "message": "VLM visual documentation was excluded by the user.",
                    "text": None,
                }
            else:
                vlm_result = {
                    "available": False,
                    "message": "VLM visual documentation was unavailable because no valid primary image existed.",
                    "text": None,
                }

            st.write("Composing the PDF with provenance, images, limitations, and verification checklist...")
            pdf_path = APP_OUTPUTS / (
                "qadamcare_comprehensive_"
                + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                + ".pdf"
            )
            generate_intelligent_pdf_report(
                output_path=pdf_path,
                result=_pdf_safe_result(result),
                llm_result=llm_result,
                vlm_result=vlm_result,
                safety_text=SAFETY_TEXT,
                model_metadata={
                    "llm_model": OLLAMA_TEXT_MODEL,
                    "vlm_model": OLLAMA_VISION_MODEL,
                },
            )
            st.session_state["comprehensive_pdf_path"] = str(pdf_path)
            status.update(label="Comprehensive PDF generated", state="complete")

    comprehensive_pdf = st.session_state.get("comprehensive_pdf_path")
    if comprehensive_pdf and Path(comprehensive_pdf).exists():
        st.success("The comprehensive AI-integrated report is ready.")
        st.download_button(
            "Download comprehensive intelligent PDF",
            data=Path(comprehensive_pdf).read_bytes(),
            file_name=Path(comprehensive_pdf).name,
            mime="application/pdf",
            use_container_width=True,
        )
        st.caption(
            "The PDF separates user-entered history, validated model outputs, LLM reasoning, and VLM visual interpretation."
        )
