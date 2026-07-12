from datetime import datetime


def yes_no(value):
    return "Present" if value else "Not reported"


def build_clinical_report(
    rgb_result,
    clinical_inputs,
    clinical_summary,
    advanced_ai,
    thermal_result=None,
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    visual_status = (
        "Visible ulcer-like region detected"
        if rgb_result.get("area_pixels", 0) > 0
        else "No visible ulcer-like region detected by the RGB model"
    )

    area_text = (
        f"{rgb_result.get('area_pixels', 0):,.0f} pixels"
    )

    size_info = advanced_ai.get("size_estimation", {})

    if size_info.get("area_cm2") is not None:
        area_text += f" ({size_info['area_cm2']:.2f} cm², calibrated)"
    else:
        area_text += " (physical size unavailable; calibration required)"

    clinical_flags = []

    flag_map = {
        "redness": "Redness reported",
        "swelling": "Swelling reported",
        "warmth": "Warmth reported",
        "discharge": "Discharge or odour reported",
        "fever": "Systemic symptoms or fever reported",
        "neuropathy": "Neuropathy history reported",
        "vascular_disease": "Peripheral vascular disease history reported",
        "probe_to_bone": "Probe-to-bone finding reported",
    }

    for key, text in flag_map.items():
        if clinical_inputs.get(key):
            clinical_flags.append(text)

    if not clinical_flags:
        clinical_flags.append("No additional clinician-entered warning findings recorded")

    thermal_section = "No thermal image submitted."

    if thermal_result and thermal_result.get("thermal_available"):
        thermal_section = (
            f"Thermal classifier output: {thermal_result['predicted_pattern']}. "
            f"Dataset-defined DM Group pattern probability: "
            f"{thermal_result['dm_probability']:.1%}. "
            f"Decision threshold: {thermal_result['threshold']:.2f}. "
            "This result is not a diagnosis of diabetes, infection, ulcer severity, "
            "or a measured temperature abnormality."
        )

    report = {
        "report_time": now,
        "title": "QadamCare AI — Clinical Decision-Support Summary",
        "purpose": (
            "This report provides AI-assisted image documentation and structured "
            "clinical-review support. It does not replace clinician examination, "
            "diagnosis, or treatment decisions."
        ),
        "image_assessment": [
            f"Image quality status: {rgb_result.get('image_quality', 'Not available')}",
            f"RGB model result: {visual_status}",
            f"Segmentation confidence: {rgb_result.get('confidence', 0):.1%}",
            f"Estimated visible-region area: {area_text}",
        ],
        "clinical_findings": [
            f"Pain score entered: {clinical_inputs.get('pain_level', 0)}/10",
            f"Redness: {yes_no(clinical_inputs.get('redness'))}",
            f"Swelling: {yes_no(clinical_inputs.get('swelling'))}",
            f"Warmth: {yes_no(clinical_inputs.get('warmth'))}",
            f"Discharge/odour: {yes_no(clinical_inputs.get('discharge'))}",
            f"Systemic symptoms/fever: {yes_no(clinical_inputs.get('fever'))}",
            f"Neuropathy history: {yes_no(clinical_inputs.get('neuropathy'))}",
            f"Peripheral vascular disease history: {yes_no(clinical_inputs.get('vascular_disease'))}",
            f"Probe-to-bone finding: {yes_no(clinical_inputs.get('probe_to_bone'))}",
        ],
        "review_flags": clinical_flags,
        "prototype_interpretation": {
            "clinical_concern": clinical_summary.get("clinical_concern", "Not available"),
            "review_priority": advanced_ai.get("review_priority", "Not available"),
            "infection_review_flag": advanced_ai.get("infection_suspicion", {}).get(
                "level",
                "Not available",
            ),
            "note": (
                "The infection-review flag is a rule-based prompt for clinician "
                "assessment; it does not confirm infection."
            ),
        },
        "thermal_section": thermal_section,
        "recommended_next_step": (
            "Clinician review is required. Confirm wound depth, infection status, "
            "vascular status, neuropathy status, offloading needs, and formal "
            "classification through standard clinical assessment."
        ),
        "limitations": [
            "RGB segmentation identifies visible ulcer-like regions only.",
            "Image-based analysis cannot determine deep tissue involvement, osteomyelitis, or infection confirmation.",
            "Physical wound area requires a calibrated image capture method.",
            "Thermal output is limited to the dataset-defined classifier task and is not a diagnostic thermography result.",
            "All AI outputs require clinician verification before use in care decisions.",
        ],
    }

    return report


def report_to_markdown(report):
    lines = [
        f"# {report['title']}",
        f"**Generated:** {report['report_time']}",
        "",
        "## Purpose",
        report["purpose"],
        "",
        "## Image Assessment",
    ]

    for item in report["image_assessment"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Clinician-Entered Findings"])

    for item in report["clinical_findings"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Review Flags"])

    for item in report["review_flags"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Prototype Interpretation",
        f"- Clinical concern level: {report['prototype_interpretation']['clinical_concern']}",
        f"- Review priority: {report['prototype_interpretation']['review_priority']}",
        f"- Infection-review flag: {report['prototype_interpretation']['infection_review_flag']}",
        f"- Note: {report['prototype_interpretation']['note']}",
        "",
        "## Thermal Module",
        report["thermal_section"],
        "",
        "## Recommended Next Step",
        report["recommended_next_step"],
        "",
        "## Limitations",
    ])

    for item in report["limitations"]:
        lines.append(f"- {item}")

    return "\n".join(lines)
