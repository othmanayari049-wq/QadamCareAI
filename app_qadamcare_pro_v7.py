"""QadamCare professional v7 entry point.

Repairs the v6 wrapper's nested triple-quoted replacement strings before
compiling it. This preserves all v6 STANDUP safety and ROI improvements.

Run with:
    python -m streamlit run app_qadamcare_pro_v7.py
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
V6_APP = ROOT / "app_qadamcare_pro_v6.py"
source = V6_APP.read_text(encoding="utf-8")

replacements = {
    "standup_upload_replacement = '''elif workflow == WORKFLOW_STANDUP:":
        'standup_upload_replacement = """elif workflow == WORKFLOW_STANDUP:',
    '    st.markdown("### Required input")\'\'\'\nsource, count = standup_upload_pattern.subn':
        '    st.markdown("### Required input")"""\nsource, count = standup_upload_pattern.subn',
    "standup_execution_replacement = '''        elif workflow == WORKFLOW_STANDUP:":
        'standup_execution_replacement = """        elif workflow == WORKFLOW_STANDUP:',
    '            file = uploaded.get("pseudo_thermal")\'\'\'\nsource, count = standup_execution_pattern.subn':
        '            file = uploaded.get("pseudo_thermal")"""\nsource, count = standup_execution_pattern.subn',
}

for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(
            "Expected v6 string marker was not found. Pull the latest branch and retry."
        )
    source = source.replace(old, new, 1)

# Validate the corrected v6 wrapper before executing it.
compiled = compile(source, str(V6_APP), "exec")
namespace = {"__file__": str(V6_APP), "__name__": "__main__"}
exec(compiled, namespace)
