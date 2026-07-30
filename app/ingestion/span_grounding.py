"""
Span grounding: the mechanism that makes "every claim traces to a source"
real rather than aspirational.

The failure mode this prevents: a model paraphrases a proof point into
existence, and a plausible-sounding number ends up on a profile with nothing
behind it. The defence is that the model never returns content - it returns
INDICES into the source text, and this module re-slices the literal substring
from those indices. The model gets no opportunity to invent.

This is the most important component in the system. If it is subtly wrong, the
core promise of the product silently breaks. It ships with a test that feeds
known text, gets indices, and asserts the re-sliced substring matches exactly.
"""

from __future__ import annotations

from app.schemas.claim import SourceSpan


class SpanGroundingError(ValueError):
    """Raised when a model-returned index pair cannot be grounded in the source."""


def ground_span(
    *,
    document_id: str,
    source_text: str,
    start_index: int,
    end_index: int,
    locator: str | None = None,
) -> SourceSpan:
    """
    Turn model-returned indices into a verified SourceSpan by re-slicing the
    literal substring from the source. The returned span's `text` is the source
    itself, never anything a model produced.

    Raises SpanGroundingError if the indices are out of bounds or inverted, so
    the caller can drop the claim to a coaching prompt instead of publishing an
    ungrounded one.
    """
    if start_index < 0 or end_index > len(source_text):
        raise SpanGroundingError(
            f"Indices [{start_index}, {end_index}) fall outside source of "
            f"length {len(source_text)}"
        )
    if end_index <= start_index:
        raise SpanGroundingError(
            f"end_index ({end_index}) must exceed start_index ({start_index})"
        )

    substring = source_text[start_index:end_index]

    # The SourceSpan validator independently re-checks that len(text) matches
    # the index range, so a bug here is caught twice.
    return SourceSpan(
        document_id=document_id,
        start_index=start_index,
        end_index=end_index,
        text=substring,
        locator=locator,
    )


def reverify_span(span: SourceSpan, current_source_text: str) -> bool:
    """
    Re-read the source and confirm the span still points at the same substring.

    The generation-time validator calls this: it re-reads the stored original
    file and checks that every span backing a generated number still resolves
    to the exact text it claimed. If a file changed or an index drifted, this
    returns False and the generated asset is rejected.
    """
    if span.end_index > len(current_source_text):
        return False
    return current_source_text[span.start_index : span.end_index] == span.text
