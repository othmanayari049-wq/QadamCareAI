"""QadamCare professional v5 entry point.

This wrapper corrects the v3 source transformation so it replaces only the clinical
input helpers and preserves the original ``build_markdown_report`` implementation.

Run with:
    python -m streamlit run app_qadamcare_pro_v5.py
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
V3_APP = ROOT / "app_qadamcare_pro_v3.py"

source = V3_APP.read_text(encoding="utf-8")

old_pattern = r'''r'def build_clinical_inputs\(\):.*?def show_validation\(validation, title\):' '''.strip()
new_pattern = r'''r'def build_clinical_inputs\(\):.*?def build_markdown_report\(data\):' '''.strip()

old_marker = '''def show_validation(validation, title):\'\'\''''
new_marker = '''def build_markdown_report(data):\'\'\''''

if old_pattern not in source:
    raise RuntimeError(
        "Expected v3 helper-regex marker not found. Pull the latest branch and retry."
    )
if old_marker not in source:
    raise RuntimeError(
        "Expected v3 replacement marker not found. Pull the latest branch and retry."
    )

source = source.replace(old_pattern, new_pattern, 1)
source = source.replace(old_marker, new_marker, 1)

# Validate the corrected v3 wrapper itself before execution.
compiled = compile(source, str(V3_APP), "exec")
namespace = {"__file__": str(V3_APP), "__name__": "__main__"}
exec(compiled, namespace)
