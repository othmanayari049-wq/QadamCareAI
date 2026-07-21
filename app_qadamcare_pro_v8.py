"""QadamCare professional v8 entry point.

Corrects the STANDUP binary-class wording so it matches the dataset labels:
participants in the diabetic group versus participants in the healthy/control
group. The model is not described as a diabetic-foot-ulcer classifier.

Run with:
    python -m streamlit run app_qadamcare_pro_v8.py
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Correct the inference function before the generated app imports it.
import standup_inference as _standup_inference

_original_analyze_dm_control_pattern = _standup_inference.analyze_dm_control_pattern


def _analyze_dm_control_pattern_with_dataset_accurate_labels(*args, **kwargs):
    result = _original_analyze_dm_control_pattern(*args, **kwargs)
    dm_score = result.get("diabetic_foot_pattern_probability")
    control_score = result.get("healthy_control_pattern_probability")

    if dm_score is not None:
        result["diabetic_group_pattern_score"] = dm_score
    if control_score is not None:
        result["healthy_control_group_pattern_score"] = control_score

    predicted = result.get("predicted_pattern")
    if predicted == "Dataset-defined diabetic-foot-like pattern":
        result["predicted_pattern"] = "Dataset-defined diabetic-participant-group image pattern"
    elif predicted == "Dataset-defined healthy/control-like pattern":
        result["predicted_pattern"] = "Dataset-defined healthy/control-group image pattern"

    if result.get("model_available"):
        result["summary"] = (
            f"The paired RGB/thermal input is closer to the {result['predicted_pattern']}. "
            "This is a dataset-group image-pattern classification, not a diabetes, "
            "diabetic-foot-ulcer, infection, or severity diagnosis."
        )

    result["safety_note"] = (
        "The STANDUP binary model distinguishes image patterns learned from participants "
        "labelled diabetic versus healthy/control in the dataset. It does not determine "
        "whether a person currently has a diabetic-foot ulcer, diagnose diabetes, estimate "
        "blood glucose, grade severity, or recommend treatment."
    )
    return result


_standup_inference.analyze_dm_control_pattern = _analyze_dm_control_pattern_with_dataset_accurate_labels

# Read v6 directly, repair its nested string markers as v7 did, then correct the
# report labels embedded in its generated STANDUP workflow.
V6_APP = ROOT / "app_qadamcare_pro_v6.py"
source = V6_APP.read_text(encoding="utf-8")

quote_repairs = {
    "standup_upload_replacement = '''elif workflow == WORKFLOW_STANDUP:":
        'standup_upload_replacement = """elif workflow == WORKFLOW_STANDUP:',
    '    st.markdown("### Required input")\'\'\'\nsource, count = standup_upload_pattern.subn':
        '    st.markdown("### Required input")"""\nsource, count = standup_upload_pattern.subn',
    "standup_execution_replacement = '''        elif workflow == WORKFLOW_STANDUP:":
        'standup_execution_replacement = """        elif workflow == WORKFLOW_STANDUP:',
    '            file = uploaded.get("pseudo_thermal")\'\'\'\nsource, count = standup_execution_pattern.subn':
        '            file = uploaded.get("pseudo_thermal")"""\nsource, count = standup_execution_pattern.subn',
}
for old, new in quote_repairs.items():
    if old not in source:
        raise RuntimeError("Expected v6 string marker was not found. Pull the latest branch and retry.")
    source = source.replace(old, new, 1)

wording_replacements = {
    "STANDUP RGB–thermal dataset-pattern similarity":
        "STANDUP RGB–thermal dataset-group image-pattern classification",
    "Closest dataset-defined pattern":
        "Closest learned dataset group",
    "Model similarity score for the dataset-defined diabetic-foot group":
        "Model score for the dataset-defined diabetic participant group",
    "Model similarity score for the dataset-defined healthy/control group":
        "Model score for the dataset-defined healthy/control participant group",
    "These complementary scores come from a binary classifier and sum to approximately 100%.":
        "These complementary classifier scores compare the two dataset groups and sum to approximately 100%.",
    "They are not disease probabilities, diagnostic confidence, severity scores, blood-glucose estimates, or treatment recommendations.":
        "They are not disease probabilities, diabetes diagnoses, diabetic-foot-ulcer diagnoses, diagnostic confidence, severity scores, blood-glucose estimates, or treatment recommendations.",
}
for old, new in wording_replacements.items():
    source = source.replace(old, new)

compiled = compile(source, str(V6_APP), "exec")
namespace = {"__file__": str(V6_APP), "__name__": "__main__"}
exec(compiled, namespace)
