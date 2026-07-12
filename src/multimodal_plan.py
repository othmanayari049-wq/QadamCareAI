MULTIMODAL_MODULES = {
    "RGB Foot Image": {
        "status": "Active prototype",
        "purpose": "Ulcer segmentation, wound area, overlay, confidence, healing comparison.",
        "validation": "Trained and evaluated on wound segmentation dataset.",
    },
    "Thermal Image": {
        "status": "Research extension",
        "purpose": "Detect abnormal heat patterns, inflammation concern, temperature asymmetry.",
        "validation": "Requires calibrated thermography dataset before real AI prediction.",
    },
    "X-ray Image": {
        "status": "Future extension",
        "purpose": "Support doctor review when bone involvement or osteomyelitis is suspected.",
        "validation": "Requires labeled diabetic-foot X-ray dataset and clinical validation.",
    },
    "Clinical Findings": {
        "status": "Active prototype",
        "purpose": "Pain, redness, swelling, warmth, discharge, fever, neuropathy, vascular disease, probe-to-bone.",
        "validation": "Clinician-entered decision-support only.",
    },
}


def get_multimodal_summary():
    return MULTIMODAL_MODULES