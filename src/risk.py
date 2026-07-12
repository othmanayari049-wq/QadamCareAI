def estimate_risk_level(total_area_pixels, number_of_lesions, confidence):
    """
    Prototype risk score based only on image-derived features.
    This is not a clinical severity diagnosis.
    """

    score = 0

    if total_area_pixels > 1000:
        score += 2
    elif total_area_pixels > 300:
        score += 1

    if number_of_lesions >= 2:
        score += 1

    if confidence >= 0.85:
        score += 1

    if score >= 3:
        risk = "HIGH"
        message = "Large or highly confident visible ulcer-like region detected. Prioritized clinical review is recommended."
    elif score >= 1:
        risk = "MODERATE"
        message = "Visible ulcer-like region detected. Clinical review is recommended."
    else:
        risk = "LOW"
        message = "No major visible ulcer-like region detected by the prototype."

    return {
        "risk_level": risk,
        "risk_score": score,
        "message": message,
    }