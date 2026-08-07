"""
CV parser: native-text PDFs only, via a swappable PdfTextExtractor backend
(app/ingestion/pdf_extractor.py) - PDF_PARSER=pypdfium2|docling picks which
one actually runs, mirroring LLM_PROVIDER and STORAGE_BACKEND. Defaults to
pypdfium2 (the free-tier implementation); Docling remains the layout-aware
production path CLAUDE.md and docs/spec.md section 2 call for, opt-in via
PDF_PARSER=docling plus requirements-production.txt.

Anything that isn't a native-text, in-scope resume - a scanned image, a
corrupt/unreadable file, a near-empty document - fails loudly with a message
the freelancer can act on, IDENTICALLY regardless of which backend is
selected: this module owns that check, neither backend does.
"""

from __future__ import annotations

import logging
from typing import Callable

from app.ingestion.extractor import (
    LLMClient,
    build_default_client,
    extract_candidate_claims,
    prepare_extraction_text,
)
from app.ingestion.pdf_extractor import CVParsingError, pdf_text_extractor
from app.schemas import Claim, SourceType
from app.storage.store import file_store

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"

# Below this many characters, treat the extraction as a scanned/unreadable
# page rather than a thin-but-real resume - real CVs clear this easily.
# Applied uniformly regardless of which PdfTextExtractor backend produced
# the text - see that protocol's docstring.
MIN_EXTRACTED_CHARS = 50


def _looks_like_pdf(cv_bytes: bytes) -> bool:
    return cv_bytes[:5] == PDF_MAGIC


def extract_pdf_text(cv_bytes: bytes) -> str:
    """
    Run the configured PdfTextExtractor (PDF_PARSER) over the PDF bytes and
    return the document's plain text. Raises CVParsingError if the backend
    can't load the file at all, or the result looks like a scan (a
    near-empty text layer).
    """
    text = pdf_text_extractor.extract_text(cv_bytes)
    if len(text.strip()) < MIN_EXTRACTED_CHARS:
        raise CVParsingError(
            "This PDF has no meaningful extractable text - it looks like a "
            "scanned image. Only native (text-based) PDFs are supported; "
            "re-export from your word processor or upload a text CV instead."
        )
    return text


def parse_cv(
    cv_bytes: bytes,
    client: LLMClient | None = None,
    on_status: Callable[..., None] | None = None,
) -> list[Claim]:
    """Parse a CV upload into grounded claims. Native PDFs only."""
    logger.info("Starting CV extraction (%d bytes)", len(cv_bytes))
    if on_status is not None:
        on_status(stage="parsing_cv", detail="Reading PDF")
    if not _looks_like_pdf(cv_bytes):
        raise CVParsingError(
            "Unsupported file type - only native PDF resumes are supported this sprint."
        )

    text = extract_pdf_text(cv_bytes)
    # Boilerplate-stripped, length-capped BEFORE storage - see
    # prepare_extraction_text's docstring for why the SAME prepared string
    # must go to both put_source and extract_candidate_claims below (never
    # the raw `text` to one and the prepared version to the other).
    extraction_text = prepare_extraction_text(text)
    # Spans are grounded against `extraction_text`, so SourceSpan.document_id
    # must be that text document's id - but the original PDF is retained too,
    # linked to it, so a parser backend switch or upgrade can always get back
    # to the untouched source rather than just its current extraction.
    pair = file_store.put_source(original=cv_bytes, text=extraction_text, original_suffix=".pdf")

    if on_status is not None:
        on_status(stage="extracting_claims", detail="Extracting claims from CV")
    llm_client = client or build_default_client()
    return extract_candidate_claims(
        llm_client,
        document_id=pair.text_id,
        text=extraction_text,
        source_type=SourceType.CV,
        locator_prefix="cv",
    )
