def generate_clinical_support_summary(quality, features, confidence, risk, visit=None):
    area = features["total_area_pixels"]
    lesions = features["number_of_lesions"]

    if quality["status"] != "PASS":
        review_priority = "RETAKE IMAGE"
        clinical_impression = "Image quality is not sufficient for reliable AI-assisted visual review."
        recommended_action = "Retake the image using better lighting, focus, and positioning before clinical interpretation."
    elif lesions == 0:
        review_priority = "LOW PRIORITY"
        clinical_impression = "No clear visible ulcer-like region was detected by the AI model."
        recommended_action = "Routine review may still be required if symptoms, neuropathy, pain, redness, swelling, discharge, or high-risk diabetic history are present."
    elif risk["risk_level"] == "HIGH":
        review_priority = "PRIORITY REVIEW"
        clinical_impression = "The AI detected a visible ulcer-like region with higher concern based on predicted area, confidence, and risk score."
        recommended_action = "Doctor or wound-care clinician review is recommended as a priority."
    elif risk["risk_level"] == "MODERATE":
        review_priority = "CLINICAL REVIEW"
        clinical_impression = "The AI detected a visible ulcer-like region that should be reviewed by a clinician."
        recommended_action = "Clinical review is recommended. The AI result should support documentation and triage only."
    else:
        review_priority = "ROUTINE REVIEW"
        clinical_impression = "The AI detected a low-concern visible finding."
        recommended_action = "Routine review may be appropriate depending on patient history and symptoms."

    visual_findings = []
    visual_findings.append(f"Image quality status: {quality['status']}.")
    visual_findings.append(f"Detected ulcer-like regions: {lesions}.")
    visual_findings.append(f"Total predicted wound area: {area} pixels.")
    visual_findings.append(f"Average AI confidence: {confidence * 100:.1f}%.")

    if lesions == 1:
        visual_findings.append("A single visible ulcer-like region was identified.")
    elif lesions > 1:
        visual_findings.append("Multiple visible ulcer-like regions were identified.")
    else:
        visual_findings.append("No visible ulcer-like region was identified by the prototype.")

    reasoning_points = []
    if quality["status"] == "PASS":
        reasoning_points.append("The image passed the technical quality gate.")
    else:
        reasoning_points.append("The image did not pass the technical quality gate, so visual interpretation may be unreliable.")

    if confidence >= 0.85:
        reasoning_points.append("The model confidence is high for the detected region.")
    elif confidence >= 0.65:
        reasoning_points.append("The model confidence is moderate and should be reviewed carefully.")
    elif lesions > 0:
        reasoning_points.append("The model confidence is relatively low, so clinician confirmation is especially important.")

    if area > 1000:
        reasoning_points.append("The predicted wound area is relatively large in pixel terms.")
    elif area > 300:
        reasoning_points.append("The predicted wound area is moderate in pixel terms.")
    elif lesions > 0:
        reasoning_points.append("The predicted wound area is small in pixel terms.")

    trend_note = "No previous visit was provided, so healing trend cannot be assessed."
    future_prediction = "Future healing cannot be estimated without serial visits."

    if visit is not None:
        trend_note = (
            f"Compared with the previous visit, the wound trend is classified as "
            f"{visit['status']} with {visit['change_percent']}% area change."
        )

        if visit["status"] == "IMPROVING":
            future_prediction = (
                "If the same decreasing trend continues, the short-term trajectory appears favorable. "
                "Continued monitoring is still recommended."
            )
        elif visit["status"] == "WORSENING":
            future_prediction = (
                "The predicted wound area increased compared with the previous visit. "
                "If this trend continues, delayed healing or deterioration may become more likely. "
                "Closer clinical follow-up is recommended."
            )
        else:
            future_prediction = (
                "The predicted wound area is relatively stable. "
                "If no improvement is seen over repeated visits, clinician reassessment may be needed."
            )

    possible_considerations = [
        "Diabetic foot ulcer or ulcer-like skin defect",
        "Pressure-related wound",
        "Traumatic skin defect",
        "Post-treatment or healing wound appearance",
        "Other dermatologic or vascular-related skin breakdown",
    ]

    infection_screening = (
        "Infection cannot be confirmed or excluded from a single RGB image. "
        "The clinician should evaluate redness, warmth, swelling, discharge, odor, pain, fever, and systemic symptoms."
    )

    missing_info = [
        "Pain level",
        "Redness, swelling, warmth, discharge, or odor",
        "Fever or systemic symptoms",
        "Diabetes duration and HbA1c if available",
        "Peripheral neuropathy status",
        "Peripheral vascular disease status",
        "Previous ulcer or amputation history",
        "Current dressing or wound treatment",
        "Offloading or footwear status",
        "Medication and antibiotic history if relevant",
    ]

    cautions = [
        "AI cannot diagnose diabetic foot ulcer from image alone.",
        "AI cannot confirm infection from image alone.",
        "AI cannot determine wound depth from a 2D photograph.",
        "AI cannot recommend medication, antibiotics, or surgical treatment.",
        "Final decision must be made by a qualified clinician.",
    ]

    follow_up = "Clinician review recommended."
    if review_priority == "LOW PRIORITY":
        follow_up = "Routine clinical review or follow-up may be sufficient if no concerning symptoms are present."
    elif review_priority == "ROUTINE REVIEW":
        follow_up = "Routine clinical review is suggested."
    elif review_priority == "CLINICAL REVIEW":
        follow_up = "Clinical review is recommended; consider follow-up imaging within 1 week depending on symptoms."
    elif review_priority == "PRIORITY REVIEW":
        follow_up = "Priority clinical review is recommended; consider wound-care referral if clinically appropriate."
    elif review_priority == "RETAKE IMAGE":
        follow_up = "Retake image before relying on AI output."

    healing_score = 0
    if quality["status"] == "PASS":
        healing_score += 20
    if confidence >= 0.85:
        healing_score += 20
    elif confidence >= 0.65:
        healing_score += 12
    if area < 300:
        healing_score += 20
    elif area < 1000:
        healing_score += 12
    else:
        healing_score += 5
    if visit is not None:
        if visit["status"] == "IMPROVING":
            healing_score += 40
        elif visit["status"] == "STABLE":
            healing_score += 25
        else:
            healing_score += 10
    else:
        healing_score += 20

    return {
        "review_priority": review_priority,
        "urgency": review_priority,
        "clinical_impression": clinical_impression,
        "recommended_action": recommended_action,
        "visual_findings": visual_findings,
        "reasoning_points": reasoning_points,
        "trend_note": trend_note,
        "future_prediction": future_prediction,
        "possible_considerations": possible_considerations,
        "infection_screening": infection_screening,
        "missing_clinical_information": missing_info,
        "clinical_cautions": cautions,
        "follow_up_suggestion": follow_up,
        "healing_score": healing_score,
    }
