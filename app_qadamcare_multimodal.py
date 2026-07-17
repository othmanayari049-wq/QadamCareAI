import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality import assess_image_quality
from ulcer_segmentation_inference import analyze_ulcer_segmentation
from thermal_risk_zones import analyze_thermal_risk_zones, save_thermal_risk_outputs
from standup_inference import analyze_dm_control_pattern, analyze_r0_r1_r2_risk_pattern
from measurement_support import build_measurement_summary

APP_OUTPUTS = ROOT / "outputs" / "app_reports"
APP_OUTPUTS.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="QadamCare AI Multimodal", page_icon="🦶", layout="wide")


def save_upload(uploaded_file, prefix):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in [".png", ".jpg", ".jpeg"]:
        suffix = ".png"
    path = APP_OUTPUTS / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"
    path.write_bytes(uploaded_file.getvalue())
    return path


def write_safety_box():
    st.warning(
        "QadamCare AI is an educational screening-support prototype. It does not diagnose diabetes, "
        "estimate blood glucose, predict exact future ulcer location, or replace clinician assessment."
    )


def build_markdown_report(patient_id, rgb_quality, dm_result, risk_result, thermal_zone_result, ulcer_result, measurement_summary):
    image_quality_score = measurement_summary["image_quality_score"]
    thermal_m = measurement_summary["thermal_measurements"]
    ulcer_m = measurement_summary["ulcer_measurements"]
    monitor = measurement_summary["monitoring_score"]

    report = []
    report.append("# QadamCare AI Multimodal Screening-Support Report")
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nPatient / Visit ID: {patient_id}")

    report.append("\n## Image Quality")
    report.append(f"- RGB quality status: {rgb_quality.get('status', 'N/A')}")
    report.append(f"- Image quality score: {image_quality_score.get('image_quality_score')}/100")
    report.append(f"- Image quality level: {image_quality_score.get('image_quality_level')}")
    report.append(f"- Reasons: {', '.join(image_quality_score.get('reasons', []))}")

    report.append("\n## RGB + Thermal STANDUP Image-Pattern Result")
    report.append(f"- Result: {dm_result.get('predicted_pattern')}")
    report.append(f"- Diabetic-foot-like probability: {dm_result.get('diabetic_foot_pattern_probability')}")
    report.append(f"- Healthy/control-like probability: {dm_result.get('healthy_control_pattern_probability')}")
    report.append(f"- Note: {dm_result.get('safety_note')}")

    report.append("\n## R0/R1/R2 Risk-Pattern Result")
    report.append(f"- Result: {risk_result.get('predicted_risk_pattern')}")
    report.append(f"- Probabilities: {risk_result.get('probabilities')}")
    report.append("- Status: experimental only; not used as the main final monitoring output.")
    report.append(f"- Note: {risk_result.get('safety_note')}")

    report.append("\n## Thermal Measurement and Risk-Zone Highlighting")
    report.append(f"- Overall monitoring zone from thermal overlay: {thermal_zone_result.get('overall_monitoring_zone')}")
    report.append(f"- Hotspot ratio: {thermal_m.get('hotspot_ratio_percent')}%")
    report.append(f"- Medium-zone ratio: {thermal_m.get('medium_zone_ratio_percent')}%")
    report.append(f"- Thermal asymmetry score: {thermal_m.get('asymmetry_score')}")
    report.append(f"- Dominant high-monitoring zone: {thermal_m.get('dominant_high_monitoring_zone')}")
    report.append(f"- Mean relative intensity: {thermal_m.get('mean_relative_intensity')}")
    report.append(f"- Thermal contrast: {thermal_m.get('thermal_contrast')}")
    report.append(f"- Concern points: {', '.join(thermal_m.get('concern_points', []))}")
    report.append(f"- Note: {thermal_zone_result.get('safety_note')}")

    report.append("\n## Visible Ulcer-Like Segmentation and Measurements")
    report.append(f"- Result: {ulcer_result.get('visible_ulcer_like_region_detected')}")
    report.append(f"- Number of regions: {ulcer_result.get('number_of_regions')}")
    report.append(f"- Total area pixels: {ulcer_m.get('area_pixels')}")
    report.append(f"- Area percent of analyzed image: {ulcer_m.get('area_percent_of_image')}")
    report.append(f"- Largest region area: {ulcer_m.get('largest_region_area_pixels')}")
    report.append(f"- Dominant ulcer-like zone: {ulcer_m.get('dominant_ulcer_zone')}")
    report.append(f"- Shape irregularity proxy: {ulcer_m.get('shape_irregularity')}")
    report.append(f"- Note: {ulcer_result.get('safety_note')}")

    report.append("\n## Combined QadamCare Monitoring Score")
    report.append(f"- Monitoring score: {monitor.get('monitoring_score')}/100")
    report.append(f"- Monitoring level: {monitor.get('monitoring_level')}")
    report.append("- Main reasons:")
    for reason in monitor.get("main_reasons", []):
        report.append(f"  - {reason}")
    report.append(f"- Area follow-up: {monitor.get('followup_area_change', {}).get('message')}")
    report.append(f"- Hotspot follow-up: {monitor.get('followup_hotspot_change', {}).get('message')}")

    report.append("\n## Final Interpretation")
    report.append(
        "This report highlights image patterns, measurement-derived monitoring zones, and visible wound-like regions for clinician review. "
        "It does not confirm diabetes, glucose level, future ulcer location, infection, ischemia, ulcer severity, or treatment plan."
    )
    report.append(f"\nSafety note: {measurement_summary.get('safety_note')}")
    return "\n".join(report)


st.title("QadamCare AI — Multimodal RGB + Thermal Workflow")
st.caption("RGB/thermal image-pattern classification, thermal monitoring zones, measurement support, and optional visible ulcer segmentation.")
write_safety_box()

with st.sidebar:
    st.header("Visit information")
    patient_id = st.text_input("Patient / Visit ID", value="QA-DEMO-001")
    st.markdown("### Required inputs")
    st.write("1. RGB plantar-foot image")
    st.write("2. Matching thermal plantar-foot image")
    st.markdown("### Optional")
    st.write("Use the RGB image for visible ulcer-like segmentation when a wound is visible.")
    st.markdown("### Optional previous visit comparison")
    use_previous = st.checkbox("Add previous visit measurement values", value=False)
    previous_area_pixels = None
    previous_hotspot_ratio = None
    if use_previous:
        previous_area_value = st.number_input("Previous visible ulcer-like area in pixels", min_value=0.0, value=0.0, step=100.0)
        previous_hotspot_percent = st.number_input("Previous thermal hotspot ratio (%)", min_value=0.0, value=0.0, step=0.1)
        previous_area_pixels = previous_area_value if previous_area_value > 0 else None
        previous_hotspot_ratio = previous_hotspot_percent / 100.0 if previous_hotspot_percent > 0 else None

rgb_file = st.file_uploader("Upload RGB plantar-foot image", type=["png", "jpg", "jpeg"])
thermal_file = st.file_uploader("Upload matching thermal plantar-foot image", type=["png", "jpg", "jpeg"])
run_ulcer_segmentation = st.checkbox("Run visible ulcer-like segmentation on RGB image", value=True)

if rgb_file is None or thermal_file is None:
    st.info("Upload both RGB and thermal images to run the multimodal workflow.")
else:
    if st.button("Run full multimodal analysis", type="primary", use_container_width=True):
        rgb_path = save_upload(rgb_file, "rgb")
        thermal_path = save_upload(thermal_file, "thermal")

        with st.spinner("Running image-quality check..."):
            rgb_quality = assess_image_quality(rgb_path)

        with st.spinner("Running STANDUP RGB+thermal image-pattern model..."):
            dm_result = analyze_dm_control_pattern(rgb_path, thermal_path)
            risk_result = analyze_r0_r1_r2_risk_pattern(rgb_path, thermal_path)

        with st.spinner("Creating thermal hotspot/risk-zone map..."):
            thermal_zone_result = analyze_thermal_risk_zones(thermal_path)
            thermal_saved = save_thermal_risk_outputs(thermal_zone_result, APP_OUTPUTS)

        if run_ulcer_segmentation:
            with st.spinner("Running visible ulcer-like segmentation..."):
                ulcer_result = analyze_ulcer_segmentation(rgb_path)
        else:
            ulcer_result = {
                "model_available": False,
                "visible_ulcer_like_region_detected": None,
                "number_of_regions": None,
                "total_area_pixels": None,
                "confidence": None,
                "summary": "Ulcer segmentation was not run.",
                "safety_note": "No ulcer segmentation output is available for this report.",
            }

        measurement_summary = build_measurement_summary(
            rgb_quality=rgb_quality,
            dm_result=dm_result,
            thermal_result=thermal_zone_result,
            ulcer_result=ulcer_result,
            previous_area_pixels=previous_area_pixels,
            previous_hotspot_ratio=previous_hotspot_ratio,
        )

        st.session_state["multimodal_result"] = {
            "patient_id": patient_id,
            "rgb_path": str(rgb_path),
            "thermal_path": str(thermal_path),
            "rgb_quality": rgb_quality,
            "dm_result": dm_result,
            "risk_result": risk_result,
            "thermal_zone_result": thermal_zone_result,
            "thermal_saved": thermal_saved,
            "ulcer_result": ulcer_result,
            "measurement_summary": measurement_summary,
        }
        st.success("Multimodal analysis completed.")

result = st.session_state.get("multimodal_result")
if result:
    st.divider()
    st.header("1. Executive Summary")
    dm_result = result["dm_result"]
    risk_result = result["risk_result"]
    thermal_zone_result = result["thermal_zone_result"]
    ulcer_result = result["ulcer_result"]
    rgb_quality = result["rgb_quality"]
    measurement_summary = result["measurement_summary"]
    monitor = measurement_summary["monitoring_score"]
    quality_score = measurement_summary["image_quality_score"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RGB Quality", rgb_quality.get("status", "N/A"))
    c2.metric("Quality Score", f"{quality_score.get('image_quality_score')}/100")
    c3.metric("DM/Control Pattern", dm_result.get("predicted_pattern", "N/A"))
    c4.metric("Thermal Zone", thermal_zone_result.get("overall_monitoring_zone", "N/A"))
    c5.metric("Monitoring Score", f"{monitor.get('monitoring_score')}/100")

    st.header("2. RGB + Thermal STANDUP Pattern Model")
    st.write(dm_result.get("summary"))
    if not dm_result.get("model_available"):
        st.warning(dm_result.get("summary"))
    else:
        st.progress(float(dm_result.get("diabetic_foot_pattern_probability", 0.0)))
    st.caption(dm_result.get("safety_note"))

    st.header("3. R0/R1/R2 Risk-Pattern Model — Experimental Only")
    st.write(risk_result.get("summary"))
    if risk_result.get("probabilities"):
        st.json(risk_result.get("probabilities"))
        st.warning("This R0/R1/R2 result is experimental and should not be presented as reliable risk staging.")
    else:
        st.warning(risk_result.get("summary"))
    st.caption(risk_result.get("safety_note"))

    st.header("4. Thermal Hotspot / Monitoring-Zone Map")
    thermal_m = measurement_summary["thermal_measurements"]
    z1, z2, z3, z4 = st.columns(4)
    z1.metric("Overall zone", thermal_zone_result["overall_monitoring_zone"])
    z2.metric("Hotspot ratio", f"{thermal_m['hotspot_ratio_percent']}%")
    z3.metric("Asymmetry score", thermal_m["asymmetry_score"])
    z4.metric("Dominant zone", thermal_m["dominant_high_monitoring_zone"])
    st.image(thermal_zone_result["risk_zone_overlay"], caption="Thermal high/medium monitoring-zone overlay", use_container_width=True)
    if thermal_zone_result["high_monitoring_regions"]:
        st.dataframe(thermal_zone_result["high_monitoring_regions"], use_container_width=True)
    with st.expander("Thermal measurement details"):
        st.json(thermal_m)
    st.caption(thermal_zone_result["safety_note"])

    st.header("5. Optional Visible Ulcer-Like Segmentation")
    st.write(ulcer_result.get("summary"))
    if ulcer_result.get("model_available"):
        u1, u2, u3 = st.columns(3)
        u1.image(ulcer_result["image"], caption="RGB input", use_container_width=True)
        u2.image(ulcer_result["mask"] * 255, caption="Predicted mask", use_container_width=True)
        u3.image(ulcer_result["overlay"], caption="Overlay", use_container_width=True)
        with st.expander("Ulcer measurement details"):
            st.json(measurement_summary["ulcer_measurements"])
    else:
        st.warning(ulcer_result.get("summary"))
    st.caption(ulcer_result.get("safety_note"))

    st.header("6. Combined QadamCare Monitoring Score")
    m1, m2 = st.columns(2)
    m1.metric("Monitoring score", f"{monitor.get('monitoring_score')}/100")
    m2.metric("Monitoring level", monitor.get("monitoring_level"))
    st.markdown("**Main reasons:**")
    for reason in monitor.get("main_reasons", []):
        st.write("-", reason)
    st.info(monitor.get("followup_area_change", {}).get("message"))
    st.info(monitor.get("followup_hotspot_change", {}).get("message"))
    st.caption(monitor.get("safety_note"))

    st.header("7. Report")
    markdown_report = build_markdown_report(
        result["patient_id"], rgb_quality, dm_result, risk_result, thermal_zone_result, ulcer_result, measurement_summary
    )
    st.markdown(markdown_report)
    st.download_button(
        "Download Markdown report",
        data=markdown_report.encode("utf-8"),
        file_name=f"qadamcare_multimodal_report_{result['patient_id']}.md",
        mime="text/markdown",
    )
