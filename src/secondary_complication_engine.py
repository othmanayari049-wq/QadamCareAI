def _yes(value):
    return bool(value)


def _level_from_score(score):
    if score >= 6:
        return "High"
    if score >= 3:
        return "Moderate"
    if score >= 1:
        return "Low"
    return "Minimal"


def _escalation_from_flags(infection_level, vascular_level, bone_level, delayed_level):
    high_count = sum(
        level == "High"
        for level in [infection_level, vascular_level, bone_level, delayed_level]
    )

    moderate_count = sum(
        level == "Moderate"
        for level in [infection_level, vascular_level, bone_level, delayed_level]
    )

    if bone_level == "High" or high_count >= 2:
        return "Urgent clinician review suggested"

    if high_count == 1 or moderate_count >= 2:
        return "Close follow-up suggested"

    if moderate_count == 1:
        return "Clinician review recommended"

    return "Routine monitoring support"


def analyze_complication_pathways(
    rgb_result,
    clinical_inputs,
    thermal_result=None,
    previous_area_pixels=None,
):
    """
    Rule-based secondary complication pathway support.

    This is NOT diagnosis.
    It estimates which review pathway may need clinician attention.
    """

    reasons = []

    current_area = 0
    if isinstance(rgb_result, dict):
        current_area = rgb_result.get("area_pixels", 0) or rgb_result.get("predicted_area_pixels", 0) or 0

    pain_level = clinical_inputs.get("pain_level", 0)

    redness = _yes(clinical_inputs.get("redness"))
    swelling = _yes(clinical_inputs.get("swelling"))
    warmth = _yes(clinical_inputs.get("warmth"))
    discharge = _yes(clinical_inputs.get("discharge"))
    fever = _yes(clinical_inputs.get("fever"))
    neuropathy = _yes(clinical_inputs.get("neuropathy"))
    vascular_disease = _yes(clinical_inputs.get("vascular_disease"))
    probe_to_bone = _yes(clinical_inputs.get("probe_to_bone"))

    area_change_percent = None
    area_increased = False

    if previous_area_pixels is not None and previous_area_pixels > 0 and current_area > 0:
        area_change_percent = ((current_area - previous_area_pixels) / previous_area_pixels) * 100
        area_increased = area_change_percent > 10

    # -----------------------------
    # Infection-review pathway
    # -----------------------------
    infection_score = 0
    infection_reasons = []

    if redness:
        infection_score += 1
        infection_reasons.append("redness was reported")
    if swelling:
        infection_score += 1
        infection_reasons.append("swelling was reported")
    if warmth:
        infection_score += 1
        infection_reasons.append("warmth was reported")
    if discharge:
        infection_score += 2
        infection_reasons.append("discharge or odor was reported")
    if fever:
        infection_score += 2
        infection_reasons.append("fever or systemic symptoms were reported")
    if pain_level >= 7:
        infection_score += 1
        infection_reasons.append("high pain level was reported")
    if area_increased:
        infection_score += 1
        infection_reasons.append("visible wound-like area increased compared with previous entry")

    infection_level = _level_from_score(infection_score)

    # -----------------------------
    # Vascular-review pathway
    # -----------------------------
    vascular_score = 0
    vascular_reasons = []

    if vascular_disease:
        vascular_score += 3
        vascular_reasons.append("vascular disease was reported")
    if area_increased:
        vascular_score += 1
        vascular_reasons.append("wound-like area increased compared with previous entry")
    if current_area > 25000:
        vascular_score += 1
        vascular_reasons.append("large visible wound-like area was detected")
    if thermal_result is not None and isinstance(thermal_result, dict):
        thermal_label = str(thermal_result.get("predicted_label", "")).lower()
        thermal_concern = str(thermal_result.get("thermal_concern", "")).lower()
        if "dm" in thermal_label or thermal_concern in ["moderate", "high"]:
            vascular_score += 1
            vascular_reasons.append("thermal-pattern module suggested a review signal")

    vascular_level = _level_from_score(vascular_score)

    # -----------------------------
    # Delayed-healing pathway
    # -----------------------------
    delayed_score = 0
    delayed_reasons = []

    if vascular_disease:
        delayed_score += 2
        delayed_reasons.append("vascular disease may increase delayed-healing review priority")
    if neuropathy:
        delayed_score += 2
        delayed_reasons.append("neuropathy was reported")
    if current_area > 25000:
        delayed_score += 1
        delayed_reasons.append("large wound-like area was detected")
    if area_increased:
        delayed_score += 2
        delayed_reasons.append("wound-like area increased compared with previous entry")
    if infection_level in ["Moderate", "High"]:
        delayed_score += 1
        delayed_reasons.append("infection-review signals may increase follow-up priority")

    delayed_level = _level_from_score(delayed_score)

    # -----------------------------
    # Bone-involvement review pathway
    # -----------------------------
    bone_score = 0
    bone_reasons = []

    if probe_to_bone:
        bone_score += 6
        bone_reasons.append("probe-to-bone concern was reported")
    if discharge and pain_level >= 7:
        bone_score += 1
        bone_reasons.append("discharge with high pain was reported")

    bone_level = _level_from_score(bone_score)

    if bone_score == 0:
        bone_pathway = "Not enough information"
    else:
        bone_pathway = bone_level

    # -----------------------------
    # Primary pathway selection
    # -----------------------------
    pathway_scores = {
        "Infection-review pathway": infection_score,
        "Vascular-review pathway": vascular_score,
        "Delayed-healing pathway": delayed_score,
        "Bone-involvement review pathway": bone_score,
    }

    primary_pathway = max(pathway_scores, key=pathway_scores.get)

    if pathway_scores[primary_pathway] == 0:
        primary_pathway = "No dominant secondary pathway detected"

    secondary_pathways = [
        name
        for name, score in sorted(pathway_scores.items(), key=lambda item: item[1], reverse=True)
        if score > 0 and name != primary_pathway
    ]

    escalation_priority = _escalation_from_flags(
        infection_level=infection_level,
        vascular_level=vascular_level,
        bone_level=bone_level,
        delayed_level=delayed_level,
    )

    reasons.extend(infection_reasons)
    reasons.extend(vascular_reasons)
    reasons.extend(delayed_reasons)
    reasons.extend(bone_reasons)

    if not reasons:
        reasons.append("No major clinician-entered secondary complication signals were reported.")

    return {
        "module_name": "Complication Pathway Prediction Support",
        "primary_pathway": primary_pathway,
        "secondary_pathways": secondary_pathways[:2],
        "infection_review_flag": infection_level,
        "vascular_review_flag": vascular_level,
        "delayed_healing_flag": delayed_level,
        "bone_involvement_review_flag": bone_pathway,
        "escalation_priority": escalation_priority,
        "area_change_percent": area_change_percent,
        "reasons": reasons,
        "safety_note": (
            "This is a rule-based secondary complication review-support module. "
            "It does not diagnose infection, ischemia, osteomyelitis, ulcer depth, or amputation risk. "
            "Final interpretation requires clinician assessment."
        ),
    }
