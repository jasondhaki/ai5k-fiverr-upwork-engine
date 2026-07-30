"""
CV parser tests. Native PDFs only, via Docling: a non-PDF upload fails loudly
before Docling is even invoked, and a PDF with no real text layer (a stand-in
for a scanned page) fails loudly rather than silently producing zero claims.
"""

from __future__ import annotations

import json

import pytest
from fpdf import FPDF

from app.ingestion.cv_parser import CVParsingError, extract_pdf_text, parse_cv
from app.schemas import SourceType
from app.storage.store import file_store

pytest.importorskip("docling", reason="Docling not installed; see requirements.txt")

_SCANNED_TEXT = (
    "Built a RAG system over 100k documents for a legal-tech client and "
    "cut retrieval latency 40 percent by rewriting the chunking strategy."
)


class _FakeLLMClient:
    def __init__(self, body: str) -> None:
        self.body = body

    def complete(self, *, system: str, prompt: str) -> str:
        return self.body


def _build_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    if text:
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, text)
    return bytes(pdf.output())


def _build_scanned_pdf(text: str) -> bytes:
    """
    A genuine stand-in for a scanned CV: the words exist only as pixels in a
    rasterized image, with no PDF text object at all - unlike _build_pdf(""),
    which is just an empty page. Rendering real, substantial text into the
    image (rather than a blank square) makes this a real assertion about OCR
    being off: if RapidOCR were still enabled, it would read this text back,
    clear MIN_EXTRACTED_CHARS, and parse_cv would NOT raise. With OCR
    disabled, only the (nonexistent) native text layer is read, so it must.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 300), color="white")
    draw = ImageDraw.Draw(image)
    draw.multiline_text((20, 20), text, fill="black", spacing=12)

    pdf = FPDF()
    pdf.add_page()
    pdf.image(image, x=10, y=10, w=190)
    return bytes(pdf.output())


def test_parse_cv_rejects_non_pdf_bytes():
    with pytest.raises(CVParsingError):
        parse_cv(b"This is just plain text, not a PDF.")


@pytest.mark.slow
def test_parse_cv_fails_loudly_on_a_textless_pdf():
    # a structurally valid PDF with no content stream - stands in for a scan
    blank_pdf = _build_pdf("")
    with pytest.raises(CVParsingError):
        parse_cv(blank_pdf)


@pytest.mark.slow
def test_parse_cv_fails_loudly_on_an_image_only_scanned_pdf():
    """
    The real regression this guards against: Docling enables OCR by default,
    which would read the words back out of the image, clear
    MIN_EXTRACTED_CHARS, and let a scanned CV through as if it were native
    text - silently defeating fail-loudly-on-scans. With OCR disabled in
    cv_parser's converter config, this must raise CVParsingError for the
    "no meaningful extractable text" reason specifically, not some other
    Docling failure.
    """
    scanned_pdf = _build_scanned_pdf(_SCANNED_TEXT)

    with pytest.raises(CVParsingError, match="no meaningful extractable text"):
        parse_cv(scanned_pdf)


@pytest.mark.slow
def test_parse_cv_extracts_and_grounds_claims_from_a_native_pdf():
    resume_text = "Built a RAG system over 100k documents for a legal-tech client."
    pdf_bytes = _build_pdf(resume_text)

    # Ground against whatever Docling actually extracts, rather than assuming
    # its exact formatting matches the input verbatim.
    extracted = extract_pdf_text(pdf_bytes)
    body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "built a RAG system",
                    "skill_ids": ["rag_systems"],
                    "evidence_quote": "Built a RAG system",
                }
            ]
        }
    )

    claims = parse_cv(pdf_bytes, client=_FakeLLMClient(body))

    published = [c for c in claims if c.publishable]
    assert published
    assert published[0].source_type == SourceType.CV
    assert published[0].source_span.text == "Built a RAG system"

    # The span's document_id must resolve to the extracted text (what the
    # indices are actually valid against), and that text document must link
    # back to the original PDF bytes we uploaded - not just its extraction.
    text_id = published[0].source_span.document_id
    assert file_store.get_text(text_id) == extracted
    original_id = file_store.get_original_id(text_id)
    assert file_store.get_bytes(original_id) == pdf_bytes
