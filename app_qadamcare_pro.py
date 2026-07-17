from datetime import datetime
from pathlib import Path

# Execute the validated unified application first. This preserves the full sidebar,
# workflow routing, model validation, Markdown/PDF downloads, and LLM/VLM controls.
from app_unified import *  # noqa: F401,F403

from intelligent_pdf_report import generate_intelligent_pdf_report
from local_ollama import OLLAMA_TEXT_MODEL, OLLAMA_VISION_MODEL


result = st.session_state.get("unified_result")

if result:
    st.divider()
    st.markdown("## Comprehensive AI-Integrated PDF")
    st.info(
        "This report can combine the validated analysis, all sidebar findings, images and overlays, "
        "the evidence-informed LLM report, and the VLM visual note in one structured PDF."
    )

    current_llm = st.session_state.get("llm_result")
    current_vlm = st.session_state.get("vlm_result")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Validated analysis",
        "Included" if result.get("status") else "Unavailable",
    )
    c2.metric(
        "LLM reasoning",
        "Ready" if current_llm and current_llm.get("available") else "Not generated",
    )
    c3.metric(
        "VLM visual note",
        "Ready" if current_vlm and current_vlm.get("available") else "Not generated",
    )

    include_llm = st.checkbox(
        "Include evidence-informed LLM clinical reasoning",
        value=True,
        help="If it has not already been generated, QadamCare will generate it before building the PDF.",
    )
    include_vlm = st.checkbox(
        "Include VLM visual documentation note",
        value=True,
        help="Requires a valid primary image. If missing, the PDF will explain that the VLM section was unavailable.",
    )

    if st.button(
        "Generate comprehensive intelligent PDF",
        type="primary",
        use_container_width=True,
    ):
        with st.status("Building comprehensive report...", expanded=True) as status:
            llm_result = st.session_state.get("llm_result")
            vlm_result = st.session_state.get("vlm_result")

            if include_llm and not (llm_result and llm_result.get("available")):
                st.write("Generating evidence-informed LLM reasoning...")
                llm_result = polish_report_with_llm(result["markdown"])
                st.session_state["llm_result"] = llm_result

            primary_path = result.get("primary_image_path")
            primary_exists = primary_path and Path(primary_path).exists()
            if include_vlm and primary_exists and not (vlm_result and vlm_result.get("available")):
                st.write("Generating VLM visual documentation note...")
                vlm_result = build_multimodal_summary(
                    structured_report_markdown=result["markdown"],
                    rgb_image_path=primary_path,
                    rgb_overlay_path=result.get("overlay_path"),
                )
                st.session_state["vlm_result"] = vlm_result

            if not include_llm:
                llm_result = {
                    "available": False,
                    "message": "LLM reasoning was excluded by the user.",
                    "text": None,
                }
            if not include_vlm:
                vlm_result = {
                    "available": False,
                    "message": "VLM visual documentation was excluded by the user.",
                    "text": None,
                }
            elif not primary_exists:
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
                result=result,
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
            "The PDF separates user-entered facts, validated model outputs, LLM reasoning, and VLM visual interpretation."
        )
