"""
PDF text extraction with PyMuPDF.

Handles valid, empty, unreadable (corrupted / encrypted) and image-only PDFs.
OCR is NOT implemented; scanned PDFs are reported clearly instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Union

from config import get_logger

log = get_logger("pdf_extractor")

NO_TEXT_MESSAGE = (
    "No readable text was detected. This prototype currently supports text-based PDFs."
)


class PDFExtractionError(Exception):
    """User-facing extraction error (message is safe to display)."""


@dataclass
class ExtractedDocument:
    text: str
    page_count: int
    pages: List[str] = field(default_factory=list)
    image_only_pages: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


def _clean_page_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[ \u00a0]{2,}", " ", text)
    # Re-join words hyphenated across line breaks: "agree-\nment" -> "agreement"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    # Drop bare page numbers such as "3" or "Page 3 of 10"
    lines = [ln for ln in lines if not re.fullmatch(r"\s*(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?\s*", ln, re.I)]
    return "\n".join(lines).strip()


def extract_text_from_pdf(source: Union[bytes, str]) -> ExtractedDocument:
    """Extract text from PDF bytes or a file path.

    Raises PDFExtractionError with a user-friendly message on failure.
    """
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:
        try:
            import fitz  # older PyMuPDF releases
        except ImportError as exc:  # pragma: no cover
            raise PDFExtractionError("PyMuPDF is not installed. Run: pip install pymupdf") from exc

    if isinstance(source, (bytes, bytearray)):
        if len(source) == 0:
            raise PDFExtractionError("The uploaded file is empty.")
        if not bytes(source[:1024]).lstrip().startswith(b"%PDF"):
            raise PDFExtractionError("The uploaded file does not appear to be a valid PDF.")
        open_kwargs = {"stream": bytes(source), "filetype": "pdf"}
    else:
        open_kwargs = {"filename": str(source)}

    try:
        doc = fitz.open(**open_kwargs)
    except Exception as exc:
        log.warning("PyMuPDF could not open document: %s", exc)
        raise PDFExtractionError("The PDF could not be opened. It may be corrupted or not a real PDF file.") from exc

    try:
        if doc.needs_pass:
            raise PDFExtractionError("The PDF is password-protected. Please upload an unlocked copy.")
        if doc.page_count == 0:
            raise PDFExtractionError("The PDF has no pages.")
        pages, image_only = [], 0
        for page in doc:
            try:
                raw = page.get_text("text", sort=True)
            except Exception as exc:
                log.warning("Failed to read page %s: %s", page.number, exc)
                raw = ""
            cleaned = _clean_page_text(raw)
            if not cleaned and page.get_images(full=False):
                image_only += 1
            pages.append(cleaned)
        meta = {k: v for k, v in (doc.metadata or {}).items() if v}
        page_count = doc.page_count
    finally:
        doc.close()

    text = "\n\n".join(p for p in pages if p).strip()
    if len(re.sub(r"\W", "", text)) < 20:
        raise PDFExtractionError(NO_TEXT_MESSAGE)
    return ExtractedDocument(text=text, page_count=page_count, pages=pages, image_only_pages=image_only,
                             metadata=meta)
