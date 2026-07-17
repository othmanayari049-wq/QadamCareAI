from pathlib import Path

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


NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1D4ED8")
LIGHT_BLUE = colors.HexColor("#EAF2FF")
RED = colors.HexColor("#991B1B")
LIGHT_RED = colors.HexColor("#FEE2E2")
GRAY = colors.HexColor("#6B7280")
LIGHT_GRAY = colors.HexColor("#F3F4F6")
BORDER = colors.HexColor("#CBD5E1")


def _safe(value, default="Not provided"):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "UnifiedTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "UnifiedSubtitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "UnifiedH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "UnifiedBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.5,
            textColor=colors.black,
            spaceAfter=3,
        ),
        "small": ParagraphStyle(
            "UnifiedSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10.5,
            textColor=GRAY,
        ),
        "alert": ParagraphStyle(
            "UnifiedAlert",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=RED,
        ),
        "cell": ParagraphStyle(
            "UnifiedCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=10.5,
        ),
    }


def _footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(BORDER)
    canvas.line(1.5 * cm, 1.25 * cm, width - 1.5 * cm, 1.25 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(1.5 * cm, 0.82 * cm, "QadamCare AI — screening-support documentation only")
    canvas.drawRightString(width - 1.5 * cm, 0.82 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _box(title, body, styles, border=BLUE, fill=LIGHT_BLUE):
    table = Table(
        [
            [Paragraph(f"<b>{_safe(title)}</b>", styles["body"])],
            [Paragraph(_safe(body), styles["body"])],
        ],
        colWidths=[17 * cm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 0.8, border),
                ("LINEBELOW", (0, 0), (-1, 0), 0.4, border),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _kv_table(rows, styles):
    data = [
        [
            Paragraph(f"<b>{_safe(key)}</b>", styles["cell"]),
            Paragraph(_safe(value), styles["cell"]),
        ]
        for key, value in rows
    ]
    table = Table(data, colWidths=[5 * cm, 12 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_GRAY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
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
        pair = valid[start : start + 2]
        cells = []
        for path, caption in pair:
            img = Image(str(path), width=7.3 * cm, height=7.3 * cm, kind="proportional")
            cells.append(
                Table(
                    [[img], [Paragraph(_safe(caption), styles["small"])]],
                    colWidths=[7.7 * cm],
                )
            )
        while len(cells) < 2:
            cells.append(Paragraph("", styles["small"]))
        rows.append(cells)

    table = Table(rows, colWidths=[8.2 * cm, 8.2 * cm], hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def generate_unified_pdf_report(
    output_path,
    title,
    workflow,
    status,
    patient,
    sections,
    images=None,
    safety_text=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.3 * cm,
        bottomMargin=1.6 * cm,
        title=_safe(title),
        author="QadamCare AI",
    )

    story = [
        Paragraph(_safe(title), styles["title"]),
        Paragraph(_safe(workflow), styles["subtitle"]),
        _box(
            "Clinical safety statement",
            _safe(safety_text),
            styles,
            border=RED,
            fill=LIGHT_RED,
        ),
        Spacer(1, 0.25 * cm),
        Paragraph("Case and execution summary", styles["h1"]),
        _kv_table(
            [
                ("Patient ID", patient.get("id")),
                ("Visit ID", patient.get("visit_id")),
                ("Patient name", patient.get("name")),
                ("Age", patient.get("age")),
                ("Gender", patient.get("gender")),
                ("Diabetes type", patient.get("diabetes_type")),
                ("Selected workflow", workflow),
                ("Execution status", status),
            ],
            styles,
        ),
        Spacer(1, 0.25 * cm),
    ]

    grid = _image_grid(images, styles)
    if grid is not None:
        story.extend(
            [
                Paragraph("Submitted and generated images", styles["h1"]),
                grid,
                PageBreak(),
            ]
        )

    for index, section in enumerate(sections or [], start=1):
        heading = section.get("heading", f"Section {index}")
        lines = section.get("lines", [])
        block = [Paragraph(f"{index}. {_safe(heading)}", styles["h1"])]
        if not lines:
            block.append(Paragraph("No information available.", styles["small"]))
        else:
            for line in lines:
                if line:
                    block.append(Paragraph(f"• {_safe(line)}", styles["body"]))
        block.append(Spacer(1, 0.15 * cm))
        story.append(KeepTogether(block))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return str(output_path)
