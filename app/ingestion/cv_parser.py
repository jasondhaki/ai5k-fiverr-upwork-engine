"""
CV parser: native-text PDFs only, via Docling.

Docling handles layout-aware text extraction for a handful of common
single-column resume layouts. Anything it can't pull real text from - a
scanned image, a password-protected file, a near-empty document - fails
loudly with a message the freelancer can act on. This sprint does not chase
OCR or two-column layouts with photos in them.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.ingestion.extractor import LLMClient, build_default_client, extract_candidate_claims
from app.schemas import Claim, SourceType
from app.storage.store import file_store

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"

# Below this many characters, treat the extraction as a scanned/unreadable
# page rather than a thin-but-real resume - real CVs clear this easily.
MIN_EXTRACTED_CHARS = 50


class CVParsingError(ValueError):
    """Raised when a CV can't be parsed. Never swallowed - the caller must
    surface this to the freelancer rather than silently returning zero claims."""


def _looks_like_pdf(cv_bytes: bytes) -> bool:
    return cv_bytes[:5] == PDF_MAGIC


def _build_converter():
    """
    Docling enables OCR (RapidOCR) by default, which would silently defeat
    the whole point of the near-empty-text check below: a scanned page would
    get OCR'd into real-looking text, clear MIN_EXTRACTED_CHARS, and never
    raise CVParsingError. OCR is out of scope this slice - only the native
    text layer is read, so a scan reliably comes back near-empty and fails
    loudly instead of silently "working". This also skips downloading and
    loading the OCR model weights, which is most of Docling's cold-start cost.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(do_ocr=False)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def extract_pdf_text(cv_bytes: bytes) -> str:
    """
    Run Docling's converter over the PDF bytes and return the document's
    plain text. Raises CVParsingError if Docling can't load the file or the
    result looks like a scan (a near-empty text layer).
    """
    try:
        converter = _build_converter()
    except ImportError as exc:
        raise CVParsingError(
            "Docling is not installed. Run `pip install docling` "
            "(see requirements.txt) to enable CV parsing."
        ) from exc

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "cv.pdf"
        tmp_path.write_bytes(cv_bytes)
        try:
            result = converter.convert(str(tmp_path))
        except Exception as exc:  # Docling raises a variety of its own errors
            raise CVParsingError(f"Docling could not parse this PDF: {exc}") from exc

    text = result.document.export_to_text()
    if len(text.strip()) < MIN_EXTRACTED_CHARS:
        raise CVParsingError(
            "This PDF has no meaningful extractable text - it looks like a "
            "scanned image. Only native (text-based) PDFs are supported; "
            "re-export from your word processor or upload a text CV instead."
        )
    return text


def parse_cv(cv_bytes: bytes, client: LLMClient | None = None) -> list[Claim]:
    """Parse a CV upload into grounded claims. Native PDFs only."""
    logger.info("Starting CV extraction (%d bytes)", len(cv_bytes))
    if not _looks_like_pdf(cv_bytes):
        raise CVParsingError(
            "Unsupported file type - only native PDF resumes are supported this sprint."
        )

    text = extract_pdf_text(cv_bytes)
    # Spans are grounded against `text`, so SourceSpan.document_id must be the
    # text document's id - but the original PDF is retained too, linked to it,
    # so a Docling upgrade or a "show me page 2" request can always get back
    # to the untouched source rather than just its current extraction.
    pair = file_store.put_source(original=cv_bytes, text=text, original_suffix=".pdf")

    llm_client = client or build_default_client()
    return extract_candidate_claims(
        llm_client,
        document_id=pair.text_id,
        text=text,
        source_type=SourceType.CV,
        locator_prefix="cv",
    )
