def estimate_anatomical_location(lesion, image_shape=(256, 256)):
    h, w = image_shape
    cx, cy = lesion.get("centroid", [w / 2, h / 2])

    x_ratio = cx / w
    y_ratio = cy / h

    if y_ratio < 0.25:
        region = "Toe / forefoot region"
    elif y_ratio < 0.55:
        region = "Midfoot region"
    else:
        region = "Heel / rearfoot region"

    if x_ratio < 0.33:
        side = "left-side image region"
    elif x_ratio > 0.67:
        side = "right-side image region"
    else:
        side = "central image region"

    return f"{region}, {side}"


def estimate_real_world_size(area_pixels, pixels_per_cm=None):
    if pixels_per_cm is None or pixels_per_cm <= 0:
        return {
            "area_cm2": None,
            "note": "Real-world area not available because no calibration scale was provided."
        }

    area_cm2 = area_pixels / (pixels_per_cm ** 2)

    return {
        "area_cm2": round(area_cm2, 2),
        "note": "Estimated using the user-provided pixels-per-centimeter calibration value."
    }


def infection_suspicion_score(clinical_inputs, visit=None):
    score = 0
    reasons = []

    if clinical_inputs.get("redness"):
        score += 2
        reasons.append("Redness reported")

    if clinical_inputs.get("warmth"):
        score += 2
        reasons.append("Warmth reported")

    if clinical_inputs.get("swelling"):
        score += 2
        reasons.append("Swelling reported")

    if clinical_inputs.get("discharge"):
        score += 3
        reasons.append("Discharge or odor reported")

    if clinical_inputs.get("fever"):
        score += 3
        reasons.append("Fever or systemic symptoms reported")

    if visit is not None and visit.get("status") == "WORSENING":
        score += 2
        reasons.append("Wound area increased compared with previous visit")

    if score >= 7:
        level = "HIGH"
    elif score >= 3:
        level = "MODERATE"
    elif score > 0:
        level = "LOW"
    else:
        level = "NOT INDICATED"

    return {
        "score": score,
        "level": level,
        "reasons": reasons,
        "note": "This is not an infection diagnosis. Infection must be assessed clinically."
    }


def estimate_wagner_grade(features, clinical_inputs):
    lesions = features["number_of_lesions"]

    if lesions == 0:
        grade = "No visible ulcer-like region detected"
        explanation = "The model did not detect a visible ulcer-like region."
    elif clinical_inputs.get("probe_to_bone"):
        grade = "Possible Wagner Grade 3 concern"
        explanation = "Positive probe-to-bone was entered. Clinician assessment for deeper involvement is required."
    else:
        grade = "Possible Wagner Grade 1 visual pattern"
        explanation = "A superficial visible ulcer-like region is detected, but depth and infection cannot be determined from RGB image alone."

    return {
        "estimated_grade": grade,
        "explanation": explanation,
        "caution": "This is only a screening-support estimate, not a clinical Wagner diagnosis."
    }


def diabetes_risk_modifier(diabetes_type):
    if diabetes_type == "Type I":
        return "Type I diabetes selected. Long-term duration and neuropathy/vascular status should be reviewed."
    if diabetes_type == "Type II":
        return "Type II diabetes selected. This is commonly associated with neuropathy and vascular risk factors that may affect wound healing."
    if diabetes_type == "Gestational":
        return "Gestational diabetes selected. Diabetic-foot ulcer risk should be interpreted carefully with clinician context."
    return "Diabetes type not specified. Clinician should confirm diabetes type and duration."


def advanced_clinical_analysis(
    features,
    confidence,
    risk,
    clinical_inputs,
    diabetes_type,
    visit=None,
    pixels_per_cm=None,
    image_shape=(256, 256),
):
    area_pixels = features["total_area_pixels"]

    size = estimate_real_world_size(area_pixels, pixels_per_cm)
    infection = infection_suspicion_score(clinical_inputs, visit)
    wagner = estimate_wagner_grade(features, clinical_inputs)
    diabetes_note = diabetes_risk_modifier(diabetes_type)

    lesion_locations = []
    for lesion in features["lesions"]:
        lesion_locations.append({
            "lesion_id": lesion["lesion_id"],
            "location": estimate_anatomical_location(lesion, image_shape),
            "area_pixels": lesion["area_pixels"],
            "confidence": lesion["mean_confidence"],
        })

    explainability_points = []

    if features["number_of_lesions"] > 0:
        explainability_points.append("The AI focused on the visible ulcer-like region shown in the segmentation overlay.")

    if confidence >= 0.85:
        explainability_points.append("High confidence increased the reliability of the detected region.")
    elif confidence >= 0.65:
        explainability_points.append("Moderate confidence means the result should be reviewed carefully.")
    else:
        explainability_points.append("Low confidence means clinician confirmation is especially important.")

    if visit is not None:
        explainability_points.append(f"Visit trend contributed to the interpretation: {visit['status']}.")

    if infection["level"] in ["MODERATE", "HIGH"]:
        explainability_points.append("Clinician-entered inflammatory/systemic findings increased infection concern.")

    final_summary = (
        f"The system detected {features['number_of_lesions']} ulcer-like region(s) "
        f"with total predicted area of {area_pixels} pixels and average confidence of {confidence * 100:.1f}%. "
        f"The current image-based risk level is {risk['risk_level']}. "
        f"Infection suspicion based on clinician-entered findings is {infection['level']}. "
        f"{diabetes_note} "
        "This output supports clinician review and documentation only."
    )

    return {
        "size_estimation": size,
        "infection_suspicion": infection,
        "wagner_estimate": wagner,
        "diabetes_note": diabetes_note,
        "lesion_locations": lesion_locations,
        "explainability_points": explainability_points,
        "final_summary": final_summary,
    }
