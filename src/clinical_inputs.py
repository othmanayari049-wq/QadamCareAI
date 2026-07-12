def summarize_clinical_inputs(inputs):
    concern_points = []
    critical_points = []

    if inputs.get("pain_level", 0) >= 7:
        concern_points.append("High reported pain level")

    if inputs.get("redness"):
        concern_points.append("Redness reported")

    if inputs.get("swelling"):
        concern_points.append("Swelling reported")

    if inputs.get("warmth"):
        concern_points.append("Warmth reported")

    if inputs.get("discharge"):
        concern_points.append("Discharge or odor reported")
        critical_points.append("Discharge or odor may require infection review")

    if inputs.get("fever"):
        concern_points.append("Fever or systemic symptoms reported")
        critical_points.append("Systemic symptoms require clinician attention")

    if inputs.get("neuropathy"):
        concern_points.append("Known or suspected neuropathy")

    if inputs.get("vascular_disease"):
        concern_points.append("Known or suspected vascular disease")
        critical_points.append("Vascular disease may increase delayed-healing risk")

    if inputs.get("probe_to_bone"):
        concern_points.append("Positive probe-to-bone finding reported")
        critical_points.append("Positive probe-to-bone finding may require osteomyelitis workup")

    if critical_points:
        clinical_concern = "HIGH"
    elif len(concern_points) >= 4:
        clinical_concern = "HIGH"
    elif len(concern_points) >= 2:
        clinical_concern = "MODERATE"
    elif len(concern_points) == 1:
        clinical_concern = "LOW"
    else:
        clinical_concern = "NOT REPORTED"

    return {
        "clinical_concern": clinical_concern,
        "concern_points": concern_points,
        "critical_points": critical_points,
        "note": "These are clinician-entered findings. They support triage and documentation but do not replace medical assessment."
    }
