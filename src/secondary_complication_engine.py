def _state(value):
    """Normalize tri-state values to True, False, or None."""
    if value is True:
        return True
    if value is False:
        return False
    text = str(value or "").strip().lower()
    if text in {"yes", "present", "positive", "true", "1"}:
        return True
    if text in {"no", "absent", "negative", "false", "0"}:
        return False
    return None


def _level_from_score(score):
    if score >= 6:
        return "High"
    if score >= 3:
        return "Moderate"
    if score >= 1:
        return "Low"
    return "No positive signal entered"


def _escalation_from_flags(infection_level, vascular_level, bone_level, delayed_level, assessed_count):
    if assessed_count < 4:
        return "Insufficient assessed information"

    levels = [infection_level, vascular_level, bone_level, delayed_level]
    high_count = sum(level == "High" for level in levels)
    moderate_count = sum(level == "Moderate" for level in levels)

    if bone_level == "High" or high_count >= 2:
        return "Urgent clinician review suggested"
    if high_count == 1 or moderate_count >= 2:
        return "Close clinician follow-up suggested"
    if moderate_count == 1:
        return "Clinician review recommended"
    return "No escalation generated from the assessed entries"


def analyze_complication_pathways(
    rgb_result,
    clinical_inputs,
    thermal_result=None,
    previous_area_pixels=None,
):
    """Rule-based review support using only explicitly assessed findings.

    Unassessed fields remain unknown and are never converted to negative findings. This
    module does not diagnose a complication and does not infer absent findings from an
    unchecked or uncompleted form.
    """
    reasons = []

    current_area = 0
    if isinstance(rgb_result, dict):
        current_area = (
            rgb_result.get("area_pixels", 0)
            or rgb_result.get("predicted_area_pixels", 0)
            or 0
        )

    pain_assessed = bool(clinical_inputs.get("pain_assessed", True))
    pain_level = clinical_inputs.get("pain_level") if pain_assessed else None

    fields = {
        "redness": _state(clinical_inputs.get("redness")),
        "swelling": _state(clinical_inputs.get("swelling")),
        "warmth": _state(clinical_inputs.get("warmth")),
        "discharge": _state(clinical_inputs.get("discharge")),
        "fever": _state(clinical_inputs.get("fever")),
        "neuropathy": _state(clinical_inputs.get("neuropathy")),
        "vascular_disease": _state(clinical_inputs.get("vascular_disease")),
        "probe_to_bone": _state(clinical_inputs.get("probe_to_bone")),
        "periwound_callus": _state(clinical_inputs.get("periwound_callus")),
        "undermining_tunneling": _state(clinical_inputs.get("undermining_tunneling")),
    }
    assessed_count = sum(value is not None for value in fields.values()) + int(pain_assessed)
    unknown_count = sum(value is None for value in fields.values()) + int(not pain_assessed)

    area_change_percent = None
    area_increased = False
    if previous_area_pixels is not None and previous_area_pixels > 0 and current_area > 0:
        area_change_percent = ((current_area - previous_area_pixels) / previous_area_pixels) * 100
        area_increased = area_change_percent > 10

    infection_score = 0
    infection_reasons = []
    if fields["redness"] is True:
        infection_score += 1
        infection_reasons.append("redness was entered as present")
    if fields["swelling"] is True:
        infection_score += 1
        infection_reasons.append("swelling was entered as present")
    if fields["warmth"] is True:
        infection_score += 1
        infection_reasons.append("warmth was entered as present")
    if fields["discharge"] is True:
        infection_score += 2
        infection_reasons.append("discharge or odour was entered as present")
    if fields["fever"] is True:
        infection_score += 2
        infection_reasons.append("fever or systemic symptoms were entered as present")
    if pain_level is not None and pain_level >= 7:
        infection_score += 1
        infection_reasons.append("high pain was entered")
    if area_increased:
        infection_score += 1
        infection_reasons.append("visible wound-like pixel area increased by more than 10%")
    infection_level = _level_from_score(infection_score)

    vascular_score = 0
    vascular_reasons = []
    if fields["vascular_disease"] is True:
        vascular_score += 3
        vascular_reasons.append("vascular disease was entered as present")
    if area_increased:
        vascular_score += 1
        vascular_reasons.append("visible wound-like pixel area increased")
    if current_area > 25000:
        vascular_score += 1
        vascular_reasons.append("a large model-space wound-like pixel area was detected")
    if thermal_result is not None and isinstance(thermal_result, dict):
        thermal_concern = str(thermal_result.get("thermal_concern", "")).lower()
        if thermal_concern in {"moderate", "high"}:
            vascular_score += 1
            vascular_reasons.append("the relative thermal module generated a review signal")
    vascular_level = _level_from_score(vascular_score)

    delayed_score = 0
    delayed_reasons = []
    if fields["vascular_disease"] is True:
        delayed_score += 2
        delayed_reasons.append("vascular disease was entered as present")
    if fields["neuropathy"] is True:
        delayed_score += 2
        delayed_reasons.append("neuropathy was entered as present")
    if fields["periwound_callus"] is True:
        delayed_score += 1
        delayed_reasons.append("periwound callus was entered as present")
    if current_area > 25000:
        delayed_score += 1
        delayed_reasons.append("a large model-space wound-like pixel area was detected")
    if area_increased:
        delayed_score += 2
        delayed_reasons.append("visible wound-like pixel area increased")
    if infection_level in {"Moderate", "High"}:
        delayed_score += 1
        delayed_reasons.append("infection-review signals increased follow-up priority")
    delayed_level = _level_from_score(delayed_score)

    bone_score = 0
    bone_reasons = []
    if fields["probe_to_bone"] is True:
        bone_score += 6
        bone_reasons.append("a positive probe-to-bone finding was entered")
    if fields["undermining_tunneling"] is True:
        bone_score += 1
        bone_reasons.append("undermining or tunnelling was entered as present")
    if fields["discharge"] is True and pain_level is not None and pain_level >= 7:
        bone_score += 1
        bone_reasons.append("discharge with high pain was entered")
    bone_level = _level_from_score(bone_score)
    bone_pathway = "Not assessed / insufficient information" if fields["probe_to_bone"] is None else bone_level

    pathway_scores = {
        "Infection-review pathway": infection_score,
        "Vascular-review pathway": vascular_score,
        "Delayed-healing pathway": delayed_score,
        "Bone-involvement review pathway": bone_score,
    }
    primary_pathway = max(pathway_scores, key=pathway_scores.get)
    if pathway_scores[primary_pathway] == 0:
        primary_pathway = (
            "Insufficient assessed information"
            if assessed_count < 4
            else "No positive secondary signal entered"
        )

    secondary_pathways = [
        name
        for name, score in sorted(pathway_scores.items(), key=lambda item: item[1], reverse=True)
        if score > 0 and name != primary_pathway
    ]

    escalation_priority = _escalation_from_flags(
        infection_level,
        vascular_level,
        bone_level,
        delayed_level,
        assessed_count,
    )

    reasons.extend(infection_reasons)
    reasons.extend(vascular_reasons)
    reasons.extend(delayed_reasons)
    reasons.extend(bone_reasons)
    if unknown_count:
        reasons.append(
            f"{unknown_count} clinical field(s) were not assessed and were not treated as negative findings."
        )
    if not reasons:
        reasons.append(
            "No positive secondary signals were entered among the assessed fields. This does not establish that complications are absent."
        )

    return {
        "module_name": "Rule-based complication review support",
        "primary_pathway": primary_pathway,
        "secondary_pathways": secondary_pathways[:2],
        "infection_review_flag": infection_level,
        "vascular_review_flag": vascular_level,
        "delayed_healing_flag": delayed_level,
        "bone_involvement_review_flag": bone_pathway,
        "escalation_priority": escalation_priority,
        "area_change_percent": area_change_percent,
        "assessed_field_count": assessed_count,
        "unassessed_field_count": unknown_count,
        "reasons": reasons,
        "safety_note": (
            "This deterministic module uses only explicitly entered findings. It does not diagnose infection, ischemia, osteomyelitis, wound depth, or amputation risk, and an unassessed field is never interpreted as absent. Final interpretation requires clinician assessment."
        ),
    }
