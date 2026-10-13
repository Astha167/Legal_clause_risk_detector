"""Create assets/sample_rental_agreement.pdf (fictional) for testing uploads.

Usage: python assets/make_sample_pdf.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer  # noqa: E402

from services.mock_data import SAMPLE_RENTAL_TEXT  # noqa: E402


def build(path: Path) -> Path:
    ss = getSampleStyleSheet()
    story = []
    for block in SAMPLE_RENTAL_TEXT.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) == 1 and lines[0].isupper():
            story.append(Paragraph(lines[0], ss["Title"]))
            continue
        if lines[0][:1].isdigit():
            story.append(Paragraph(f"<b>{lines[0]}</b>", ss["Heading4"]))
            lines = lines[1:]
        for ln in lines:
            story.append(Paragraph(ln, ss["BodyText"]))
        story.append(Spacer(1, 6))
    SimpleDocTemplate(str(path), pagesize=A4, title="Sample Rental Agreement").build(story)
    return path


if __name__ == "__main__":
    out = build(Path(__file__).resolve().parent / "sample_rental_agreement.pdf")
    print(f"Created {out}")
