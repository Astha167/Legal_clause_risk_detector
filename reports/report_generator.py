"""
Professional PDF report of an analysis result (ReportLab).

Sections: contract information, analysis summary, risk distribution, overall
screening score, high/medium/low-risk clauses (with risk factors and model
explanations), document observations and the disclaimer.
"""
from __future__ import annotations

import io
import unicodedata
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import config

INK = colors.HexColor("#1C2536")
SLATE = colors.HexColor("#5A6477")
RULE = colors.HexColor("#E2E5EB")
SEAL = colors.HexColor("#2F4B7C")
RISK = {"High": colors.HexColor("#B83A33"), "Medium": colors.HexColor("#B7790B"), "Low": colors.HexColor("#2E7D55")}
TINT = {"High": colors.HexColor("#FBECEA"), "Medium": colors.HexColor("#FCF3DF"), "Low": colors.HexColor("#E7F3EC")}

_REPLACE = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
            "\u2026": "...", "\u20b9": "Rs.", "\u2022": "-", "\u00a0": " ", "\u2212": "-"}


class ReportError(Exception):
    pass


def _t(text) -> str:
    """Escape for Paragraph markup and map characters the built-in fonts cannot draw."""
    s = "" if text is None else str(text)
    for k, v in _REPLACE.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch if ord(ch) < 256 else "?" for ch in s if not unicodedata.combining(ch))
    return escape(s)


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Times-Bold", fontSize=22, textColor=INK,
                                alignment=TA_LEFT, spaceAfter=4),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=10, textColor=SLATE, spaceAfter=12),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=15, textColor=INK,
                             spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, textColor=INK,
                             spaceBefore=4, spaceAfter=3),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=13.5,
                               textColor=INK),
        "clause": ParagraphStyle("c", parent=ss["Normal"], fontName="Times-Roman", fontSize=10, leading=14,
                                 textColor=INK),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontName="Helvetica", fontSize=8.5, leading=11.5,
                                textColor=SLATE),
        "bullet": ParagraphStyle("bu", parent=ss["Normal"], fontName="Helvetica", fontSize=9.3, leading=13,
                                 leftIndent=10, bulletIndent=2, textColor=INK),
    }


def _kv_table(rows, widths=(45 * mm, 125 * mm)):
    t = Table([[Paragraph(f"<b>{_t(k)}</b>", _styles()["body"]), Paragraph(_t(v), _styles()["body"])]
               for k, v in rows], colWidths=widths)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5)]))
    return t


def _pie(summary):
    d = Drawing(170, 130)
    p = Pie()
    p.x, p.y, p.width, p.height = 15, 10, 105, 105
    vals = [summary.get(l, 0) for l in config.LABELS]
    if sum(vals) == 0:
        vals = [1, 0, 0]
    p.data = vals
    p.labels = [f"{l} ({summary.get(l, 0)})" for l in config.LABELS]
    p.simpleLabels = 1
    p.sideLabels = 1
    for i, l in enumerate(config.LABELS):
        p.slices[i].fillColor = RISK[l]
        p.slices[i].strokeColor = colors.white
        p.slices[i].fontName = "Helvetica"
        p.slices[i].fontSize = 8
    d.add(p)
    d.add(String(0, 0, "", fontSize=1))
    return d


def _clause_block(c, st, engine_is_demo):
    conf = "n/a (Demo Mode)" if c.get("confidence") is None else f"{c['confidence'] * 100:.0f}%"
    heading = c.get("heading") or ""
    head = f"Clause {c['clause_id']}: {c['clause_type']}" + (
        f" ({heading})" if heading and heading.lower() != c["clause_type"].lower() else "")
    items = [
        Paragraph(_t(head), st["h3"]),
        Paragraph(f"<font color='{RISK[c['risk_label']].hexval().replace('0x', '#')}'><b>{c['risk_label'].upper()} RISK</b></font>"
                  f" &nbsp; Screening score {c['risk_score']:.0f}/100 &nbsp; Confidence {_t(conf)}", st["small"]),
        Spacer(1, 3),
    ]
    text = c["clause_text"] if len(c["clause_text"]) < 1800 else c["clause_text"][:1800] + " ..."
    box = Table([[Paragraph(_t(text), st["clause"])]], colWidths=[170 * mm])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT[c["risk_label"]]),
                             ("LINEBEFORE", (0, 0), (0, -1), 2.5, RISK[c["risk_label"]]),
                             ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                             ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    items.append(box)
    factors = [f["name"] for f in c.get("factors", []) if f["weight"] > 0]
    if factors:
        items.append(Paragraph("<b>Risk factors:</b> " + _t(", ".join(factors)), st["body"]))
    for b in c.get("explanation_bullets", []):
        items.append(Paragraph(_t(b), st["bullet"], bulletText="-"))
    shap = c.get("shap")
    if shap and shap.get("terms"):
        terms = ", ".join(f"{t['term']} ({t['value']:+.3f})" for t in shap["terms"][:8])
        label = "Model explanation" if shap.get("available") else "Key phrases (fallback, not SHAP)"
        items.append(Paragraph(f"<b>{label}:</b> {_t(terms)}", st["small"]))
        if shap.get("available"):
            items.append(Paragraph(_t(f"{shap['message']} ({shap['method']})"), st["small"]))
    elif engine_is_demo:
        items.append(Paragraph("Model explanation: not applicable (Demo Mode uses rule-based screening only).",
                               st["small"]))
    items.append(Spacer(1, 8))
    return KeepTogether(items[:4]), items[4:]


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(SLATE)
    canvas.drawString(20 * mm, 12 * mm, f"{config.APP_NAME} - automated screening report, not legal advice")
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf_report(result: dict) -> bytes:
    try:
        st = _styles()
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                                bottomMargin=20 * mm, title="Contract Risk Screening Report",
                                author=config.APP_NAME)
        s = result["summary"]
        story = [Paragraph("Contract Risk Screening Report", st["title"]),
                 Paragraph(_t(f"{config.APP_TITLE} - generated {datetime.now():%d %B %Y, %H:%M}"), st["sub"])]
        if result.get("is_demo"):
            story.append(Paragraph("<b>DEMO MODE</b> - results come from the rule-based screening engine only; "
                                   "no machine-learning model was used.", st["body"]))
            story.append(Spacer(1, 6))
        if result.get("smoke_test"):
            story.append(Paragraph("<b>WARNING</b> - produced with a smoke-test model; results are not meaningful.",
                                   st["body"]))

        story.append(Paragraph("1. Contract information", st["h2"]))
        story.append(_kv_table([
            ("Contract", result["contract_name"]),
            ("Pages / size", f"{result.get('page_count') or '-'} page(s), {result.get('file_size', 0) / 1024:.1f} KB"),
            ("Analysed on", result.get("created_at", "")),
            ("Analysis engine", result["engine_label"]),
        ]))

        story.append(Paragraph("2. Analysis summary", st["h2"]))
        story.append(_kv_table([
            ("Total clauses", s["total"]), ("High risk", s["High"]), ("Medium risk", s["Medium"]),
            ("Low risk", s["Low"]),
        ]))

        story.append(Paragraph("3. Risk distribution and 4. Overall screening score", st["h2"]))
        score_par = Paragraph(
            f"<font size='26' name='Times-Bold'>{s['overall_score']:.0f}</font><font size='11'> / 100</font><br/><br/>"
            f"<font color='{RISK[s['overall_label']].hexval().replace('0x', '#')}'><b>{s['overall_label'].upper()} RISK</b></font><br/>"
            "<font size='8' color='#5A6477'>AI-generated screening score = 0.5 x mean clause score + "
            "0.5 x mean of the top-quartile clause scores. Not a legal measurement.</font>", st["body"])
        grid = Table([[_pie(s), score_par]], colWidths=[80 * mm, 90 * mm])
        grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(grid)

        if result.get("observations"):
            story.append(Paragraph("Document-level observations", st["h3"]))
            for o in result["observations"]:
                story.append(Paragraph(_t(o), st["bullet"], bulletText="-"))

        section_no = 5
        for label in ["High", "Medium", "Low"]:
            group = [c for c in result["clauses"] if c["risk_label"] == label]
            story.append(PageBreak() if label == "High" else Spacer(1, 6))
            story.append(Paragraph(f"{section_no}. {label}-risk clauses ({len(group)})", st["h2"]))
            section_no += 1
            if not group:
                story.append(Paragraph("None.", st["body"]))
            for c in sorted(group, key=lambda x: -x["risk_score"]):
                head, rest = _clause_block(c, st, result.get("is_demo"))
                story.append(head)
                story.extend(rest)

        story.append(Paragraph("8. Risk factors", st["h2"]))
        from collections import Counter

        fc = Counter(f["name"] for c in result["clauses"] for f in c.get("factors", []) if f["weight"] > 0)
        if fc:
            t = Table([["Risk factor", "Clauses"]] + [[_t(k), v] for k, v in fc.most_common()],
                      colWidths=[130 * mm, 40 * mm])
            t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                                   ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("TEXTCOLOR", (0, 0), (-1, -1), INK)]))
            story.append(t)
        else:
            story.append(Paragraph("No rule-based risk factors were detected.", st["body"]))

        story.append(Paragraph("9. Model explanations", st["h2"]))
        story.append(Paragraph(_t(
            "Risk levels are predicted from the clause text by the engine listed above. SHAP values, where shown, "
            "describe which terms contributed to the model's prediction; they explain model behaviour and do not "
            "establish legal validity. Rule-based explanations are simplified screening interpretations of "
            "project-defined risk indicators."), st["body"]))

        story.append(Paragraph("10. Disclaimer", st["h2"]))
        story.append(Paragraph(_t(config.DISCLAIMER), st["body"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(_t("Risk labels are project-defined screening labels, not expert legal annotations. "
                                  + config.PRIVACY_NOTICE), st["small"]))
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        return buf.getvalue()
    except Exception as exc:
        from config import get_logger

        get_logger("report").exception("PDF report generation failed")
        raise ReportError("The PDF report could not be generated. See logs/app.log for details.") from exc
