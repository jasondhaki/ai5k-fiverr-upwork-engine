"""
PDF text extraction behind a protocol - the same pattern as
app/storage/store.py's FileStore (disk vs B2) and
app/ingestion/extractor.py's LLMClient (Anthropic vs Groq): cv_parser.py
depends only on PdfTextExtractor, never on a specific backend directly, so
which one actually runs is an environment-variable choice
(PDF_PARSER=pypdfium2|docling), not a code change. See CLAUDE.md's
"Swappable backends" section for why this pattern exists three times now.

Two implementations:

- PypdfiumExtractor - the free-tier default. Plain PDF text-layer
  extraction: no layout analysis, no OCR, no ML models. Measured locally at
  ~74MB peak vs Docling's ~786MB for the same document - the difference
  between fitting a demo on Render's Starter/Free instance (512MB) and
  OOM-crashing it, which is exactly what motivated this module (see git
  history / docs/spec.md's IMPLEMENTATION NOTE at section 2 for the
  tradeoff this makes).

- DoclingExtractor - the layout-aware production path spec section 2 and
  section 10 actually call for. Needs docling/torch/opencv installed - an
  OPTIONAL dependency group (requirements-production.txt), not part of the
  default install. The `docling` import happens lazily, inside __init__,
  never at module load time, so importing this module - and therefore
  cv_parser.py, and therefore the whole app - never requires docling to be
  installed unless PDF_PARSER=docling is explicitly chosen.

PDF_PARSER is read ONCE, at import time, building a singleton exactly like
STORAGE_BACKEND builds `file_store`: an unrecognized value fails loudly
immediately, and PDF_PARSER=docling without docling installed fails loudly
at that same startup moment (see DoclingExtractor.__init__) - not on first
CV upload, when it would look like an unrelated, confusing crash on a real
user's request.
"""

from __future__ import annotations

import os
from typing import Protocol


class CVParsingError(ValueError):
    """Raised when a CV can't be parsed. Never swallowed - the caller must
    surface this to the freelancer rather than silently returning zero claims."""


class PdfTextExtractor(Protocol):
    """The only PDF-extraction contract cv_parser.py knows about."""

    def extract_text(self, pdf_bytes: bytes) -> str:
        """
        Return the PDF's text - or as much of it as this backend can find.

        Raises CVParsingError only if the file itself can't be opened at all
        (corrupt, not actually a PDF despite the extension, etc). A
        near-empty result - e.g. a scanned page with no real text layer -
        is NOT this method's call to make: cv_parser.py's MIN_EXTRACTED_CHARS
        check owns that decision, uniformly, for whichever backend produced
        the text.
        """
        ...


class PypdfiumExtractor:
    """
    Free-tier default. Reads the PDF's native text layer directly via
    pypdfium2 (already a transitive dependency of docling itself, so this
    adds nothing new to requirements.txt) - no layout model, no OCR, no
    torch.

    Loses layout awareness: pages are read in the order pypdfium2's text
    extraction reports them, not a reading-order-aware reconstruction. For
    the single-column resume layouts CLAUDE.md scopes this parser to, plain
    extraction order and visual reading order coincide; a genuinely
    multi-column layout could come back with columns interleaved. See
    docs/spec.md's IMPLEMENTATION NOTE at section 2.

    A scanned/image-only page has no PDF text objects at all, so this
    correctly returns empty text for one - the same failure signal Docling
    produces with OCR disabled, just without needing OCR disabled in the
    first place (there's no OCR here to disable).
    """

    def extract_text(self, pdf_bytes: bytes) -> str:
        import pypdfium2 as pdfium

        try:
            doc = pdfium.PdfDocument(pdf_bytes)
        except pdfium.PdfiumError as exc:
            raise CVParsingError(f"pypdfium2 could not open this PDF: {exc}") from exc

        try:
            parts = []
            for page in doc:
                textpage = page.get_textpage()
                try:
                    parts.append(textpage.get_text_range())
                finally:
                    textpage.close()
                page.close()
            return "\n".join(parts)
        finally:
            doc.close()


class DoclingExtractor:
    """
    Production path: layout-aware extraction via Docling, for the common
    single-column resume layouts spec section 2 targets. Needs
    requirements-production.txt installed (docling/torch/opencv) - the
    import happens here, lazily, in __init__, not at module load, so the
    free-tier install (and every test that doesn't specifically exercise
    this class) never needs those packages.
    """

    def __init__(self) -> None:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise RuntimeError(
                "PDF_PARSER=docling requires docling to be installed - run "
                "`pip install -r requirements-production.txt` (see that file "
                "for why it's kept separate: docling pulls in torch/opencv, "
                "~600MB of image size and ~700MB of peak memory more than "
                "this app needs by default, too much for the free-tier "
                "hosting the demo runs on)."
            ) from exc

        # OCR disabled: a scanned page would otherwise get OCR'd into
        # real-looking text and silently pass the near-empty-text check
        # cv_parser.py uses to reject scans - see that check's own comment.
        # Table-structure recognition also disabled: pure memory/CPU
        # overhead for the single-column resume layouts this parser is
        # scoped to, never exercised by an in-scope document.
        pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def extract_text(self, pdf_bytes: bytes) -> str:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "cv.pdf"
            tmp_path.write_bytes(pdf_bytes)
            try:
                result = self._converter.convert(str(tmp_path))
            except Exception as exc:  # Docling raises a variety of its own errors
                raise CVParsingError(f"Docling could not parse this PDF: {exc}") from exc
        return result.document.export_to_text()


def _build_pdf_text_extractor() -> PdfTextExtractor:
    """
    The one place a PDF-parsing backend is chosen, mirroring
    app/storage/store.py's _build_file_store and
    app/ingestion/extractor.py's build_default_client. Defaults to
    'pypdfium2' so the fast test suite - and a fresh `pip install -r
    requirements.txt` - never needs docling installed. Never silently
    guesses on an unrecognized PDF_PARSER value - that's a config mistake,
    not a routing decision, so it fails loudly instead of picking one for you.
    """
    backend = os.environ.get("PDF_PARSER", "pypdfium2").strip().lower()
    if backend == "pypdfium2":
        return PypdfiumExtractor()
    if backend == "docling":
        return DoclingExtractor()
    raise ValueError(f"Unknown PDF_PARSER '{backend}' - expected 'pypdfium2' or 'docling'.")


# The single instance cv_parser.py uses - mirrors app/storage/store.py's
# `file_store` singleton. Swap PDF_PARSER, not this line, to change backend.
pdf_text_extractor: PdfTextExtractor = _build_pdf_text_extractor()
