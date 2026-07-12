from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)


# -----------------------------------------------------------------------------
# QadamCare AI - Professional clinician-facing PDF report
# ReportLab implementation
# -----------------------------------------------------------------------------
# Safety: This report is for screening-support documentation only. It must not
# be used as a diagnosis, treatment plan, or replacement for clinician judgment.
# -----------------------------------------------------------------------------


BRAND_DARK = colors.HexColor("#102A43")
BRAND_BLUE = colors.HexColor("#1D4ED8")
BRAND_LIGHT = colors.HexColor("#EAF2FF")
GREEN = colors.HexColor("#166534")
AMBER = colors.HexColor("#92400E")
RED = colors.HexColor("#991B1B")
LIGHT_RED = colors.HexColor("#FEE2E2")
LIGHT_AMBER = colors.HexColor("#FEF3C7")
LIGHT_GREEN = colors.HexColor("#DCFCE7")
LIGHT_GRAY = colors.HexColor("#F3F4F6")
MID_GRAY = colors.HexColor("#6B7280")
BORDER = colors.HexColor("#CBD5E1")
WHITE = colors.white
BLACK = colors.black


# -----------------------------
# Safe helper functions
# -----------------------------
def _safe(value, default="Not provided"):
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    return str(value)


def _pct(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "N/A"


def _num(value, default="N/A"):
    if value is None:
        return default
    return str(value)


def _first_available(mapping, keys, default=None):
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _risk_colors(level):
    level = str(level or "").upper()
    if "HIGH" in level or "URGENT" in level or "PRIORITY" in level:
        return RED, LIGHT_RED
    if "MODERATE" in level or "CLOSE" in level:
        return AMBER, LIGHT_AMBER
    if "LOW" in level or "ROUTINE" in level or "MINIMAL" in level:
        return GREEN, LIGHT_GREEN
    return BRAND_BLUE, BRAND_LIGHT


def _paragraph_list(items, styles):
    story = []
    if not items:
        story.append(Paragraph("No items reported.", styles["SmallMuted"]))
        return story
    for item in items:
        story.append(Paragraph(f"- {_safe(item)}", styles["Body"]))
    return story


def _footer_canvas(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(BORDER)
    canvas.line(1.5 * cm, 1.3 * cm, width - 1.5 * cm, 1.3 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(1.5 * cm, 0.85 * cm, "QadamCare AI - Screening-support documentation only")
    canvas.drawRightString(width - 1.5 * cm, 0.85 * cm, f"Page {doc.page}")
    canvas.restoreState()


# -----------------------------
# Styles
# -----------------------------
def _styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["Title"] = ParagraphStyle(
        "Title",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=BRAND_DARK,
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    styles["Subtitle"] = ParagraphStyle(
        "Subtitle",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=BRAND_BLUE,
        alignment=TA_CENTER,
        spaceAfter=14,
    )

    styles["H1"] = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=BRAND_DARK,
        spaceBefore=10,
        spaceAfter=8,
    )

    styles["H2"] = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=BRAND_DARK,
        spaceBefore=6,
        spaceAfter=5,
    )

    styles["Body"] = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=BLACK,
        spaceAfter=4,
    )

    styles["Small"] = ParagraphStyle(
        "Small",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=BLACK,
    )

    styles["SmallMuted"] = ParagraphStyle(
        "SmallMuted",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=MID_GRAY,
    )

    styles["BoxText"] = ParagraphStyle(
        "BoxText",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=BLACK,
    )

    styles["AlertText"] = ParagraphStyle(
        "AlertText",
        parent=base["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=RED,
    )

    styles["TableHead"] = ParagraphStyle(
        "TableHead",
        parent=base["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10,
        textColor=WHITE,
        alignment=TA_LEFT,
    )

    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=10,
        textColor=BLACK,
    )

    return styles


# -----------------------------
# Reusable PDF components
# -----------------------------
def _section_title(text, styles):
    return Paragraph(text, styles["H1"])


def _clinical_box(title, body, styles, border_color=BRAND_BLUE, fill_color=BRAND_LIGHT):
    table = Table(
        [
            [Paragraph(f"<b>{title}</b>", styles["BoxText"])],
            [Paragraph(body, styles["BoxText"])],
        ],
        colWidths=[17.0 * cm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill_color),
                ("BOX", (0, 0), (-1, -1), 1.0, border_color),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, border_color),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _kv_table(rows, styles, col_widths=(5.0 * cm, 12.0 * cm)):
    data = []
    for key, value in rows:
        data.append([
            Paragraph(f"<b>{_safe(key)}</b>", styles["TableCell"]),
            Paragraph(_safe(value), styles["TableCell"]),
        ])

    table = Table(data, colWidths=list(col_widths), hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
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


def _data_table(headers, rows, styles, widths=None):
    data = [[Paragraph(str(h), styles["TableHead"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(_safe(cell), styles["TableCell"]) for cell in row])

    table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
                ("BACKGROUND", (0, 1), (-1, -1), WHITE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _badge_table(items, styles):
    row = []
    for label, value in items:
        border, fill = _risk_colors(value)
        cell = Table(
            [
                [Paragraph(f"<b>{label}</b>", styles["SmallMuted"])],
                [Paragraph(f"<b>{_safe(value)}</b>", styles["BoxText"])],
            ],
            colWidths=[4.0 * cm],
        )
        cell.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), fill),
                    ("BOX", (0, 0), (-1, -1), 0.8, border),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        row.append(cell)

    table = Table([row], colWidths=[4.2 * cm] * len(row), hAlign="LEFT")
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return table


def _image_pair(image_path, overlay_path, styles):
    cells = []
    for path, caption in [
        (image_path, "Original image"),
        (overlay_path, "AI overlay - screening support"),
    ]:
        if path and Path(path).exists():
            img = Image(str(path), width=7.2 * cm, height=7.2 * cm)
            cells.append([img, Paragraph(caption, styles["SmallMuted"])])
        else:
            cells.append([Paragraph("Image unavailable", styles["SmallMuted"]), Paragraph(caption, styles["SmallMuted"])])

    left = Table([[cells[0][0]], [cells[0][1]]], colWidths=[7.4 * cm])
    right = Table([[cells[1][0]], [cells[1][1]]], colWidths=[7.4 * cm])
    outer = Table([[left, right]], colWidths=[8.0 * cm, 8.0 * cm], hAlign="CENTER")
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return outer


# -----------------------------
# Main PDF generator
# -----------------------------
def generate_pdf_report(
    output_path,
    patient_id,
    visit_id,
    image_path,
    overlay_path,
    quality,
    features,
    confidence,
    risk,
    recommendation,
    visit=None,
    clinical_ai=None,
    clinical_inputs=None,
    clinical_summary=None,
    advanced_ai=None,
    secondary_complication=None,
    thermal_result=None,
    fusion_result=None,
    previous_visit_context=None,
):
    """
    Create a professional QadamCare AI clinician-facing PDF report.

    All advanced objects are optional for backward compatibility. The report is
    designed for documentation and clinician review, not diagnosis.
    """

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
        title="QadamCare AI Clinical Review Report",
        author="QadamCare AI",
    )

    story = []

    # ------------------------------------------------------------------
    # Cover / summary page
    # ------------------------------------------------------------------
    story.append(Paragraph("QadamCare AI", styles["Title"]))
    story.append(Paragraph("Diabetic Foot Screening-Support and Follow-Up Documentation Report", styles["Subtitle"]))

    story.append(
        _clinical_box(
            "Clinical safety statement",
            "This report is an educational engineering prototype for screening support and documentation. "
            "It is not a diagnosis, does not confirm infection, ischemia, osteomyelitis, ulcer depth, severity, "
            "or treatment plan, and must be reviewed by a qualified healthcare professional.",
            styles,
            border_color=RED,
            fill_color=LIGHT_RED,
        )
    )
    story.append(Spacer(1, 0.25 * cm))

    risk_level = _first_available(risk, ["risk_level"], "N/A")
    review_priority = _first_available(clinical_ai, ["review_priority", "urgency"], "Clinician review recommended")
    escalation_priority = _first_available(secondary_complication, ["escalation_priority"], "Not available")
    image_quality = _first_available(quality, ["status"], "N/A")

    story.append(
        _badge_table(
            [
                ("Image quality", image_quality),
                ("AI review priority", review_priority),
                ("Prototype risk level", risk_level),
                ("Escalation pathway", escalation_priority),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.25 * cm))

    story.append(_section_title("1. Case and Visit Summary", styles))
    story.append(
        _kv_table(
            [
                ("Generated on", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                ("Patient ID", patient_id),
                ("Visit ID", visit_id),
                ("Image quality", image_quality),
                ("Detected visible ulcer-like regions", _first_available(features, ["number_of_lesions"], "N/A")),
                ("Predicted visible region area", f"{_first_available(features, ['total_area_pixels'], 'N/A')} pixels"),
                ("Segmentation confidence", _pct(confidence)),
                ("Prototype risk level", risk_level),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.25 * cm))

    story.append(_section_title("2. Visual AI Result", styles))
    story.append(_image_pair(image_path, overlay_path, styles))
    story.append(
        Paragraph(
            "The overlay highlights AI-predicted visible ulcer-like/wound-like regions. It is not ground truth and requires clinician verification.",
            styles["SmallMuted"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))

    # ------------------------------------------------------------------
    # Clinical interpretation page
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(_section_title("3. Clinician Review Recommendation", styles))
    story.append(
        _clinical_box(
            "Recommended review action",
            _safe(recommendation),
            styles,
            border_color=BRAND_BLUE,
            fill_color=BRAND_LIGHT,
        )
    )
    story.append(Spacer(1, 0.2 * cm))

    if clinical_ai:
        story.append(_section_title("4. AI Clinical Support Summary", styles))
        story.append(
            _kv_table(
                [
                    ("Review priority", _first_available(clinical_ai, ["review_priority"], "N/A")),
                    ("Healing support score", f"{_first_available(clinical_ai, ['healing_score'], 'N/A')}/100"),
                    ("Clinical impression", _first_available(clinical_ai, ["clinical_impression"], "N/A")),
                    ("Follow-up suggestion", _first_available(clinical_ai, ["follow_up_suggestion"], "N/A")),
                    ("Trend note", _first_available(clinical_ai, ["trend_note"], "N/A")),
                ],
                styles,
            )
        )
        story.append(Spacer(1, 0.15 * cm))

    if advanced_ai:
        size_info = advanced_ai.get("size_estimation", {}) if isinstance(advanced_ai, dict) else {}
        infection_info = advanced_ai.get("infection_suspicion", {}) if isinstance(advanced_ai, dict) else {}
        wagner_info = advanced_ai.get("wagner_estimate", {}) if isinstance(advanced_ai, dict) else {}
        story.append(_section_title("5. Advanced Screening-Support Outputs", styles))
        story.append(
            _kv_table(
                [
                    ("Estimated calibrated area", f"{_safe(size_info.get('area_cm2'), 'Needs calibration')} cm2" if size_info.get("area_cm2") is not None else "Needs calibration"),
                    ("Infection review signal", _safe(infection_info.get("level"), "Not available")),
                    ("Classification support note", _safe(wagner_info.get("estimated_grade"), "Not available")),
                    ("Diabetes context", _safe(advanced_ai.get("diabetes_note"), "Not available")),
                ],
                styles,
            )
        )
        story.append(
            Paragraph(
                _safe(wagner_info.get("caution"), "Classification support is not a confirmed clinical grade."),
                styles["SmallMuted"],
            )
        )
        story.append(Spacer(1, 0.15 * cm))

    if clinical_summary or clinical_inputs:
        story.append(_section_title("6. Clinician-Entered Findings", styles))
        clinical_rows = []
        if clinical_summary:
            clinical_rows.append(("Overall entered concern", clinical_summary.get("clinical_concern", "N/A")))
            clinical_rows.append(("Summary note", clinical_summary.get("note", "N/A")))
        if clinical_inputs:
            clinical_rows.extend(
                [
                    ("Pain level", clinical_inputs.get("pain_level", "N/A")),
                    ("Redness reported", clinical_inputs.get("redness", "N/A")),
                    ("Swelling reported", clinical_inputs.get("swelling", "N/A")),
                    ("Warmth reported", clinical_inputs.get("warmth", "N/A")),
                    ("Discharge/odor reported", clinical_inputs.get("discharge", "N/A")),
                    ("Fever/systemic symptoms", clinical_inputs.get("fever", "N/A")),
                    ("Neuropathy reported", clinical_inputs.get("neuropathy", "N/A")),
                    ("Vascular disease reported", clinical_inputs.get("vascular_disease", "N/A")),
                    ("Probe-to-bone finding", clinical_inputs.get("probe_to_bone", "N/A")),
                ]
            )
        story.append(_kv_table(clinical_rows, styles))
        story.append(Spacer(1, 0.15 * cm))

    # ------------------------------------------------------------------
    # Follow-up, complication, and checklist page
    # ------------------------------------------------------------------
    story.append(PageBreak())

    story.append(_section_title("7. Secondary Complication Pathway Support", styles))
    if secondary_complication:
        story.append(
            _kv_table(
                [
                    ("Primary pathway", secondary_complication.get("primary_pathway", "N/A")),
                    ("Escalation priority", secondary_complication.get("escalation_priority", "N/A")),
                    ("Infection review flag", secondary_complication.get("infection_review_flag", "N/A")),
                    ("Vascular review flag", secondary_complication.get("vascular_review_flag", "N/A")),
                    ("Delayed-healing flag", secondary_complication.get("delayed_healing_flag", "N/A")),
                    ("Bone-involvement review flag", secondary_complication.get("bone_involvement_review_flag", "N/A")),
                    ("Area change", f"{secondary_complication.get('area_change_percent'):.1f}%" if secondary_complication.get("area_change_percent") is not None else "No previous visit data"),
                ],
                styles,
            )
        )
        story.append(Paragraph("<b>Reasons selected by the pathway module:</b>", styles["Body"]))
        story.extend(_paragraph_list(secondary_complication.get("reasons", []), styles))
        story.append(
            _clinical_box(
                "Safety note",
                _safe(secondary_complication.get("safety_note"), "This pathway output is review-support only."),
                styles,
                border_color=AMBER,
                fill_color=LIGHT_AMBER,
            )
        )
    else:
        story.append(Paragraph("Secondary complication pathway module was not included in this report.", styles["SmallMuted"]))
    story.append(Spacer(1, 0.2 * cm))

    story.append(_section_title("8. Visit-to-Visit Monitoring", styles))
    if visit:
        story.append(
            _kv_table(
                [
                    ("Trend", visit.get("status", "N/A")),
                    ("Area change", f"{visit.get('change_pixels', 'N/A')} pixels"),
                    ("Change percentage", f"{visit.get('change_percent', 'N/A')}%"),
                    ("Interpretation", visit.get("message", "N/A")),
                ],
                styles,
            )
        )
    else:
        story.append(
            _clinical_box(
                "Longitudinal data limitation",
                "No previous visit area was provided. Follow-up trend interpretation is limited to the current image and clinician-entered findings.",
                styles,
                border_color=AMBER,
                fill_color=LIGHT_AMBER,
            )
        )
    story.append(Spacer(1, 0.2 * cm))

    if thermal_result:
        story.append(_section_title("9. Thermography Research Extension", styles))
        if thermal_result.get("thermal_available") is False:
            story.append(Paragraph(_safe(thermal_result.get("note"), "Thermal image was not available or did not pass validation."), styles["Body"]))
        else:
            story.append(
                _kv_table(
                    [
                        ("Predicted thermal pattern", thermal_result.get("predicted_pattern", "N/A")),
                        ("DM group pattern probability", f"{thermal_result.get('dm_probability', 0):.1%}" if thermal_result.get("dm_probability") is not None else "N/A"),
                        ("Decision threshold", thermal_result.get("threshold", "N/A")),
                        ("Safety note", thermal_result.get("safety_note", "Thermal output is research support only.")),
                    ],
                    styles,
                )
            )
        story.append(Spacer(1, 0.15 * cm))

    story.append(_section_title("10. Clinician Assessment Checklist", styles))
    checklist = [
        "Assess visible wound location, margins, base appearance, surrounding skin, and depth by clinical examination.",
        "Assess infection-related clinical signs only by clinician examination and patient history.",
        "Assess vascular status, perfusion concerns, pulses, capillary refill, and relevant history as clinically appropriate.",
        "Assess sensation/neuropathy status and pressure/offloading factors as clinically appropriate.",
        "Verify whether the AI overlay corresponds to the true wound boundary before using measurements for follow-up documentation.",
        "Compare with prior visits only when capture angle, distance, lighting, and calibration are reasonably similar.",
    ]
    story.extend(_paragraph_list(checklist, styles))
    story.append(Spacer(1, 0.2 * cm))

    story.append(_section_title("11. Final Limitation Statement", styles))
    story.append(
        _clinical_box(
            "Decision-support limitation",
            "This PDF summarizes AI screening-support outputs and clinician-entered findings. It does not establish diagnosis, severity, infection status, ischemia status, bone involvement, or treatment plan. Clinical decisions must be made by a qualified healthcare professional.",
            styles,
            border_color=RED,
            fill_color=LIGHT_RED,
        )
    )

    doc.build(story, onFirstPage=_footer_canvas, onLaterPages=_footer_canvas)
    return output_path
