"""
Upwork parser: turns pasted profile text into grounded claims.

There is no API for this - the freelancer copies their own Upwork overview
and/or work-history text into the intake form. The pasted text IS the source
document, so every claim grounds directly against exactly what they pasted.
"""

from __future__ import annotations

from app.ingestion.extractor import LLMClient, build_default_client, extract_candidate_claims
from app.schemas import Claim, SourceType
from app.storage.store import file_store

# Below this many characters there isn't enough text for extraction to be
# meaningful - fail loudly rather than silently return zero claims.
MIN_TEXT_CHARS = 20


class UpworkParsingError(ValueError):
    """Raised when pasted Upwork text is too short/empty to extract from."""


def parse_upwork_text(text: str, client: LLMClient | None = None) -> list[Claim]:
    stripped = text.strip()
    if len(stripped) < MIN_TEXT_CHARS:
        raise UpworkParsingError(
            f"Pasted Upwork text is too short ({len(stripped)} chars) to extract claims from."
        )

    # The paste IS the source, so "original" and "extracted text" are the same
    # string - but they're still stored as two linked documents, per the
    # storage contract every source goes through, rather than a special case.
    pair = file_store.put_source(original=text.encode("utf-8"), text=text, original_suffix=".txt")
    llm_client = client or build_default_client()
    return extract_candidate_claims(
        llm_client,
        document_id=pair.text_id,
        text=text,
        source_type=SourceType.UPWORK_TEXT,
        locator_prefix="upwork paste",
    )
