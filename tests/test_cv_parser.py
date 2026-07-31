"""
CV parser tests.

Two layers:
- The "day to day" tests below exercise parse_cv/extract_pdf_text against
  whatever PDF_PARSER resolves to by default (pypdfium2, the free-tier
  backend) - fast, no docling needed, safe for routine iteration.
- The conformance section further down instantiates BOTH backends directly
  (app/ingestion/pdf_extractor.py's PypdfiumExtractor and DoclingExtractor)
  and runs the same assertions against each, proving they satisfy the
  identical PdfTextExtractor contract cv_parser.py depends on - so switching
  PDF_PARSER=docling in production never surfaces a behavior gap. The
  Docling half of those is marked `slow` (real Docling PDF conversion) and
  skips cleanly via pytest.importorskip when docling isn't installed, so it
  never breaks the free-tier suite but still runs if you've installed
  requirements-production.txt.
"""

from __future__ import annotations

import json

import pytest
from fpdf import FPDF

from app.ingestion.cv_parser import CVParsingError, extract_pdf_text, parse_cv
from app.ingestion.pdf_extractor import PdfTextExtractor, PypdfiumExtractor
from app.schemas import SourceType
from app.storage.store import file_store

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
    image (rather than a blank square) makes this a real assertion: if OCR
    were reading it back, it would clear MIN_EXTRACTED_CHARS. Neither backend
    does OCR (pypdfium2 has none at all; Docling has it explicitly disabled),
    so only the (nonexistent) native text layer is read for either.
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


def test_parse_cv_fails_loudly_on_a_textless_pdf():
    # a structurally valid PDF with no content stream - stands in for a scan
    blank_pdf = _build_pdf("")
    with pytest.raises(CVParsingError):
        parse_cv(blank_pdf)


def test_parse_cv_fails_loudly_on_an_image_only_scanned_pdf():
    """
    The real regression this guards against: a scan must never be read back
    as if it were native text and let through as if it were a real resume.
    """
    scanned_pdf = _build_scanned_pdf(_SCANNED_TEXT)

    with pytest.raises(CVParsingError, match="no meaningful extractable text"):
        parse_cv(scanned_pdf)


def test_parse_cv_extracts_and_grounds_claims_from_a_native_pdf():
    resume_text = "Built a RAG system over 100k documents for a legal-tech client."
    pdf_bytes = _build_pdf(resume_text)

    # Ground against whatever the backend actually extracts, rather than
    # assuming its exact formatting matches the input verbatim.
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


# --- Conformance: both backends satisfy the same PdfTextExtractor contract --


def _docling_extractor() -> PdfTextExtractor:
    pytest.importorskip(
        "docling", reason="Docling not installed - see requirements-production.txt"
    )
    from app.ingestion.pdf_extractor import DoclingExtractor

    return DoclingExtractor()


_BACKEND_FACTORIES = {
    "pypdfium2": PypdfiumExtractor,
    "docling": _docling_extractor,
}

_BACKEND_PARAMS = ["pypdfium2", pytest.param("docling", marks=pytest.mark.slow)]


@pytest.mark.parametrize("backend_name", _BACKEND_PARAMS)
def test_backend_extracts_native_text(backend_name):
    extractor = _BACKEND_FACTORIES[backend_name]()
    pdf_bytes = _build_pdf("Built a RAG system over 100k documents for a legal-tech client.")

    text = extractor.extract_text(pdf_bytes)

    assert "RAG system" in text


@pytest.mark.parametrize("backend_name", _BACKEND_PARAMS)
def test_backend_returns_near_empty_text_for_a_scanned_pdf(backend_name):
    """Neither backend does OCR, so both must return text too short to clear
    cv_parser.MIN_EXTRACTED_CHARS for an image-only page - this is what
    cv_parser.py's own check relies on to reject scans identically either way."""
    from app.ingestion.cv_parser import MIN_EXTRACTED_CHARS

    extractor = _BACKEND_FACTORIES[backend_name]()
    scanned_pdf = _build_scanned_pdf(_SCANNED_TEXT)

    text = extractor.extract_text(scanned_pdf)

    assert len(text.strip()) < MIN_EXTRACTED_CHARS


@pytest.mark.parametrize("backend_name", _BACKEND_PARAMS)
def test_backend_raises_cv_parsing_error_on_unreadable_bytes(backend_name):
    extractor = _BACKEND_FACTORIES[backend_name]()

    with pytest.raises(CVParsingError):
        extractor.extract_text(b"not a real pdf at all, just garbage bytes")


@pytest.mark.parametrize("backend_name", _BACKEND_PARAMS)
def test_parse_cv_rejects_a_scanned_pdf_identically_on_both_backends(backend_name, monkeypatch):
    """The end-to-end behavior CLAUDE.md requires (scans must fail loudly,
    never silently produce zero claims from misread pixels) must hold via
    parse_cv() itself, not just each backend's raw extract_text - proven by
    swapping app.ingestion.cv_parser's module-level singleton for the
    duration of this test, the same way other singletons are swapped
    elsewhere in this test suite (e.g. app.platform.api.repository)."""
    extractor = _BACKEND_FACTORIES[backend_name]()
    monkeypatch.setattr("app.ingestion.cv_parser.pdf_text_extractor", extractor)

    scanned_pdf = _build_scanned_pdf(_SCANNED_TEXT)

    with pytest.raises(CVParsingError, match="no meaningful extractable text"):
        parse_cv(scanned_pdf)
