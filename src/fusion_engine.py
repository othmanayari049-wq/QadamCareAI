def compute_fusion_decision(rgb_result, clinical_summary, advanced_ai, thermal_result=None):
    score = 0
    reasons = []

    risk = rgb_result["risk"]
    features = rgb_result["features"]
    confidence = rgb_result["confidence"]

    if risk["risk_level"] == "HIGH":
        score += 3
        reasons.append("RGB model shows high image-based risk.")
    elif risk["risk_level"] == "MODERATE":
        score += 2
        reasons.append("RGB model shows moderate image-based risk.")
    elif features["number_of_lesions"] > 0:
        score += 1
        reasons.append("RGB model detected visible ulcer-like region.")

    if confidence >= 0.85:
        score += 1
        reasons.append("Segmentation confidence is high.")

    if clinical_summary["clinical_concern"] == "HIGH":
        score += 3
        reasons.append("Clinician-entered findings show high clinical concern.")
    elif clinical_summary["clinical_concern"] == "MODERATE":
        score += 2
        reasons.append("Clinician-entered findings show moderate clinical concern.")
    elif clinical_summary["clinical_concern"] == "LOW":
        score += 1
        reasons.append("Clinician-entered findings show low clinical concern.")

    infection = advanced_ai["infection_suspicion"]
    if infection["level"] == "HIGH":
        score += 3
        reasons.append("Clinical findings suggest high infection concern.")
    elif infection["level"] == "MODERATE":
        score += 2
        reasons.append("Clinical findings suggest moderate infection concern.")
    elif infection["level"] == "LOW":
        score += 1
        reasons.append("Clinical findings suggest low infection concern.")

    if thermal_result is not None and thermal_result.get("thermal_available"):
        if thermal_result["thermal_concern"] == "HIGH":
            score += 3
            reasons.append("Thermal image shows high heat-pattern concern.")
        elif thermal_result["thermal_concern"] == "MODERATE":
            score += 2
            reasons.append("Thermal image shows moderate heat-pattern concern.")
        elif thermal_result["thermal_concern"] == "LOW":
            score += 1
            reasons.append("Thermal image shows low heat-pattern concern.")

    if score >= 8:
        overall_level = "HIGH"
        decision = "Priority clinician review is recommended."
    elif score >= 4:
        overall_level = "MODERATE"
        decision = "Clinical review is recommended."
    elif score >= 1:
        overall_level = "LOW"
        decision = "Routine review may be sufficient depending on clinical context."
    else:
        overall_level = "MINIMAL"
        decision = "No major concern detected by the prototype."

    return {
        "fusion_score": score,
        "overall_level": overall_level,
        "decision_support": decision,
        "reasons": reasons,
        "note": (
            "Fusion combines RGB model output, clinician-entered findings, "
            "advanced clinical rules, and optional thermal findings. "
            "It is decision-support only, not a diagnosis."
        ),
    }
