from datetime import datetime
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from evidence_clinical_reasoning import ADA_REFERENCE, IWGDF_IDSA_REFERENCE


NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1D4ED8")
TEAL = colors.HexColor("#0F766E")
RED = colors.HexColor("#991B1B")
AMBER = colors.HexColor("#92400E")
GRAY = colors.HexColor("#64748B")
LIGHT_BLUE = colors.HexColor("#EFF6FF")
LIGHT_TEAL = colors.HexColor("#F0FDFA")
LIGHT_RED = colors.HexColor("#FEF2F2")
LIGHT_AMBER = colors.HexColor("#FFFBEB")
LIGHT_GRAY = colors.HexColor("#F8FAFC")
BORDER = colors.HexColor("#CBD5E1")


def _safe(value, default="Not available"):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _clean(text):
    text = _safe(text, "")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "AITitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=23, leading=27, textColor=NAVY, alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "subtitle": ParagraphStyle(
            "AISubtitle", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=11, leading=14, textColor=BLUE, alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "AIH1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=15, leading=19, textColor=NAVY, spaceBefore=10, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "AIH2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11.5, leading=15, textColor=TEAL, spaceBefore=7, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "AIBody", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.1, leading=12.5, textColor=colors.black, spaceAfter=3,
        ),
        "small": ParagraphStyle(
            "AISmall", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8, leading=10.5, textColor=GRAY, spaceAfter=2,
        ),
        "cell": ParagraphStyle(
            "AICell", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.3, leading=10.5,
        ),
        "quote": ParagraphStyle(
            "AIQuote", parent=base["BodyText"], fontName="Helvetica-Oblique",
            fontSize=8.7, leading=12, textColor=colors.HexColor("#334155"),
            leftIndent=10, rightIndent=10, spaceAfter=4,
        ),
    }


def _footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(BORDER)
    canvas.line(1.4 * cm, 1.25 * cm, width - 1.4 * cm, 1.25 * cm)
    canvas.setFont("Helvetica", 7.7)
    canvas.setFillColor(GRAY)
    canvas.drawString(1.4 * cm, 0.82 * cm, "QadamCare AI — research and screening-support prototype")
    canvas.drawRightString(width - 1.4 * cm, 0.82 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _box(title, body, styles, border=BLUE, fill=LIGHT_BLUE):
    table = Table(
        [
            [Paragraph(f"<b>{_clean(title)}</b>", styles["body"])],
            [Paragraph(_clean(body), styles["body"])],
        ],
        colWidths=[17.1 * cm],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 0.8, border),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _kv_table(rows, styles):
    data = []
    for key, value in rows:
        data.append([
            Paragraph(f"<b>{_clean(key)}</b>", styles["cell"]),
            Paragraph(_clean(value), styles["cell"]),
        ])
    table = Table(data, colWidths=[5.1 * cm, 12 * cm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _image_grid(images, styles):
    valid = []
    for item in images or []:
        path = Path(item.get("path", ""))
        if path.exists():
            valid.append((path, item.get("caption", "Image")))
    if not valid:
        return None

    rows = []
    for start in range(0, len(valid), 2):
        pair = valid[start:start + 2]
        cells = []
        for path, caption in pair:
            image = Image(str(path), width=7.4 * cm, height=7.0 * cm, kind="proportional")
            cells.append(Table(
                [[image], [Paragraph(_clean(caption), styles["small"])]],
                colWidths=[7.8 * cm],
            ))
        while len(cells) < 2:
            cells.append(Paragraph("", styles["small"]))
        rows.append(cells)

    table = Table(rows, colWidths=[8.3 * cm, 8.3 * cm], hAlign="CENTER")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _markdown_flowables(markdown_text, styles):
    if not markdown_text:
        return [Paragraph("AI narrative was not generated.", styles["small"])]

    flowables = []
    for raw_line in str(markdown_text).splitlines():
        line = raw_line.strip()
        if not line:
            flowables.append(Spacer(1, 0.08 * cm))
            continue
        if line.startswith("### "):
            flowables.append(Paragraph(_clean(line[4:]), styles["h2"]))
        elif line.startswith("## "):
            flowables.append(Paragraph(_clean(line[3:]), styles["h2"]))
        elif line.startswith("# "):
            flowables.append(Paragraph(_clean(line[2:]), styles["h1"]))
        elif line.startswith(("- ", "* ")):
            flowables.append(Paragraph("• " + _clean(line[2:]), styles["body"]))
        elif re.match(r"^\d+\.\s", line):
            flowables.append(Paragraph(_clean(line), styles["body"]))
        else:
            cleaned = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", _clean(line))
            flowables.append(Paragraph(cleaned, styles["body"]))
    return flowables


def generate_intelligent_pdf_report(
    output_path,
    result,
    llm_result=None,
    vlm_result=None,
    safety_text=None,
    model_metadata=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()

    patient = result.get("patient", {})
    clinical = result.get("clinical_inputs", {})
    sections = result.get("sections", [])
    images = result.get("images", [])
    llm_text = (llm_result or {}).get("text") if (llm_result or {}).get("available") else None
    vlm_text = (vlm_result or {}).get("text") if (vlm_result or {}).get("available") else None

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=1.45 * cm, leftMargin=1.45 * cm,
        topMargin=1.25 * cm, bottomMargin=1.65 * cm,
        title="QadamCare AI Comprehensive Clinical Intelligence Report",
        author="QadamCare AI",
    )

    story = [
        Paragraph("QadamCare AI", styles["title"]),
        Paragraph("Comprehensive Clinical Intelligence & Multimodal Review Report", styles["subtitle"]),
        _box(
            "Important safety statement",
            _safe(safety_text), styles, border=RED, fill=LIGHT_RED,
        ),
        Spacer(1, 0.25 * cm),
        Paragraph("Executive case summary", styles["h1"]),
        _kv_table([
            ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Patient ID", patient.get("id")),
            ("Visit ID", patient.get("visit_id")),
            ("Patient name", patient.get("name")),
            ("Age", patient.get("age")),
            ("Gender", patient.get("gender")),
            ("Diabetes type", patient.get("diabetes_type")),
            ("Selected workflow", result.get("workflow")),
            ("Execution status", result.get("status")),
            ("LLM reasoning included", "Yes" if llm_text else "No"),
            ("VLM visual note included", "Yes" if vlm_text else "No"),
        ], styles),
        Spacer(1, 0.25 * cm),
        _box(
            "Interpretation hierarchy",
            "This report separates user-entered clinical findings, validated algorithmic outputs, local LLM reasoning, and local VLM visual description. AI narratives are supportive documentation and are not verified medical conclusions.",
            styles, border=TEAL, fill=LIGHT_TEAL,
        ),
        PageBreak(),
        Paragraph("1. Patient-entered clinical context", styles["h1"]),
        _kv_table([
            ("Pain level", f"{clinical.get('pain_level', 0)}/10"),
            ("Redness", "Reported" if clinical.get("redness") else "Not reported"),
            ("Swelling", "Reported" if clinical.get("swelling") else "Not reported"),
            ("Warmth", "Reported" if clinical.get("warmth") else "Not reported"),
            ("Discharge / odour", "Reported" if clinical.get("discharge") else "Not reported"),
            ("Fever / systemic symptoms", "Reported" if clinical.get("fever") else "Not reported"),
            ("Neuropathy", "Reported" if clinical.get("neuropathy") else "Not reported"),
            ("Vascular disease", "Reported" if clinical.get("vascular_disease") else "Not reported"),
            ("Probe-to-bone", "Reported" if clinical.get("probe_to_bone") else "Not reported"),
        ], styles),
        Spacer(1, 0.2 * cm),
        Paragraph("2. Validated workflow and algorithmic results", styles["h1"]),
    ]

    for index, section in enumerate(sections, start=1):
        block = [Paragraph(f"2.{index} {_clean(section.get('heading'))}", styles["h2"])]
        lines = [line for line in section.get("lines", []) if line]
        if not lines:
            block.append(Paragraph("No information available.", styles["small"]))
        else:
            for line in lines:
                block.append(Paragraph("• " + _clean(line), styles["body"]))
        block.append(Spacer(1, 0.08 * cm))
        story.append(KeepTogether(block))

    grid = _image_grid(images, styles)
    if grid is not None:
        story.extend([
            PageBreak(),
            Paragraph("3. Image evidence and generated visualisations", styles["h1"]),
            Paragraph(
                "Images are included for traceability. Segmentation masks, attention maps, and monitoring-zone overlays are model outputs and must not be treated as ground truth.",
                styles["body"],
            ),
            grid,
        ])

    story.extend([
        PageBreak(),
        Paragraph("4. Evidence-informed LLM clinical reasoning", styles["h1"]),
        _box(
            "LLM status",
            "Included below." if llm_text else _safe((llm_result or {}).get("message"), "Not generated before PDF creation."),
            styles, border=BLUE, fill=LIGHT_BLUE,
        ),
        Spacer(1, 0.15 * cm),
    ])
    story.extend(_markdown_flowables(llm_text, styles))

    story.extend([
        PageBreak(),
        Paragraph("5. VLM visual documentation note", styles["h1"]),
        _box(
            "VLM status",
            "Included below." if vlm_text else _safe((vlm_result or {}).get("message"), "Not generated before PDF creation."),
            styles, border=TEAL, fill=LIGHT_TEAL,
        ),
        Spacer(1, 0.15 * cm),
    ])
    story.extend(_markdown_flowables(vlm_text, styles))

    story.extend([
        PageBreak(),
        Paragraph("6. Clinical verification checklist", styles["h1"]),
        Paragraph("• Verify the history, symptom duration, diabetes duration, glycaemic-control context, medications, and relevant comorbidities.", styles["body"]),
        Paragraph("• Assess the entire foot, surrounding skin, wound depth, tissue characteristics, deformity, pressure exposure, and footwear context.", styles["body"]),
        Paragraph("• Assess protective sensation and neuropathy status using appropriate clinical methods.", styles["body"]),
        Paragraph("• Assess perfusion and vascular status clinically; image colour and thermal intensity cannot establish ischemia.", styles["body"]),
        Paragraph("• Determine whether local or systemic inflammatory findings satisfy clinical criteria for infection-related concern.", styles["body"]),
        Paragraph("• Verify whether comparison with a previous visit is technically valid under similar image-capture conditions.", styles["body"]),
        Paragraph("• Review all AI outputs against direct examination before documentation or care decisions.", styles["body"]),
        Paragraph("7. Evidence basis", styles["h1"]),
        Paragraph("• " + _clean(ADA_REFERENCE), styles["small"]),
        Paragraph("• " + _clean(IWGDF_IDSA_REFERENCE), styles["small"]),
        Spacer(1, 0.2 * cm),
        Paragraph("8. Model and data provenance", styles["h1"]),
    ])

    metadata = model_metadata or {}
    provenance_rows = [
        ("Ulcer segmentation", metadata.get("ulcer_model", "FUSeg-derived visible ulcer-like segmentation model; workflow-dependent")),
        ("STANDUP fusion", metadata.get("standup_model", "RGB + grayscale thermal dataset-pattern classifier; workflow-dependent")),
        ("Pseudo-colour thermal", metadata.get("thermal_model", "Legacy thermal-only research classifier; workflow-dependent")),
        ("Local text model", metadata.get("llm_model", "Configured Ollama text model")),
        ("Local vision-language model", metadata.get("vlm_model", "Configured Ollama vision model")),
        ("Validation status", result.get("status")),
    ]
    story.append(_kv_table(provenance_rows, styles))

    story.extend([
        Spacer(1, 0.25 * cm),
        _box(
            "Final limitation",
            "This PDF may combine deterministic measurements, trained-model outputs, rules, and generative AI text. The generative sections can be incomplete or incorrect. This document does not establish diagnosis, prognosis, severity grade, infection status, future ulcer location, or treatment plan.",
            styles, border=AMBER, fill=LIGHT_AMBER,
        ),
    ])

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return str(output_path)
