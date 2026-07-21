"""QadamCare professional v4 entry point.

This wrapper fixes the v3 runtime transformation that accidentally removed
``build_markdown_report`` while replacing the clinical-input helper functions.

Run with:
    python -m streamlit run app_qadamcare_pro_v4.py
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
V3_APP = ROOT / "app_qadamcare_pro_v3.py"

source = V3_APP.read_text(encoding="utf-8")

needle = '''def show_validation(validation, title):\'\'\''''
replacement = '''def build_markdown_report(data):
    patient = data["patient"]
    lines = [
        "# QadamCare AI Unified Screening-Support Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Workflow:** {data['workflow']}",
        f"**Execution status:** {data['status']}",
        "",
        "## Patient / Visit",
        f"- Patient ID: {patient['id']}",
        f"- Patient name: {patient['name']}",
        f"- Age: {patient['age']}",
        f"- Gender: {patient['gender']}",
        f"- User-entered diabetes history: {patient['diabetes_type']}",
        f"- Visit ID: {patient['visit_id']}",
    ]
    for section in data.get("sections", []):
        lines.extend(["", f"## {section['heading']}"])
        for item in section.get("lines", []):
            if item is not None:
                lines.append(f"- {item}")
    lines.extend(["", "## Safety limitation", SAFETY_TEXT])
    return "\\n".join(lines)


def show_validation(validation, title):\'\'\''''

if needle not in source:
    raise RuntimeError(
        "The expected v3 clinical-helper replacement marker was not found. "
        "Pull the latest branch before running v4."
    )

source = source.replace(needle, replacement, 1)

namespace = {
    "__file__": str(V3_APP),
    "__name__": "__main__",
}
exec(compile(source, str(V3_APP), "exec"), namespace)
