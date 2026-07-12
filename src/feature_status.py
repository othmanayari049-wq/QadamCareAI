FEATURE_STATUS = {
    "RGB ulcer segmentation": {
        "status": "Implemented",
        "evidence": "Trained and evaluated on Foot Ulcer Segmentation Challenge dataset.",
        "current_use": "Ulcer-like region segmentation and measurement.",
    },
    "Image quality assessment": {
        "status": "Implemented - rule-based",
        "evidence": "Uses brightness, blur, contrast, and resolution checks.",
        "current_use": "Retake recommendation before AI analysis.",
    },
    "Clinical findings panel": {
        "status": "Implemented - clinician-entered",
        "evidence": "Uses symptoms and risk flags entered by the clinician.",
        "current_use": "Supports triage and report reasoning.",
    },
    "Thermal image analysis": {
        "status": "Prototype / not validated",
        "evidence": "Rule-based thermal intensity analysis only. Requires calibrated thermography dataset for model training.",
        "current_use": "Optional exploratory screening-support only.",
    },
    "X-ray analysis": {
        "status": "Planned / not implemented",
        "evidence": "Requires diabetic-foot X-ray dataset and clinical validation.",
        "current_use": "Not used for AI prediction yet.",
    },
    "Wagner grade support": {
        "status": "Screening-support estimate",
        "evidence": "Based on visible ulcer and clinician-entered flags only.",
        "current_use": "Cannot replace clinical grading.",
    },
}