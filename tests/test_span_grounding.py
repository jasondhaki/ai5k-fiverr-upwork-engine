"""
The test that guards the core promise of the product.

Feed known text, get indices, assert the re-sliced substring matches the
original EXACTLY. If this passes, span grounding is real. If the extractor is
ever changed in a way that lets a model's paraphrase leak into a span, one of
these fails. Build this before trusting the extractor.
"""

import pytest

from app.ingestion.span_grounding import (
    SpanGroundingError,
    ground_span,
    reverify_span,
)

SOURCE = "Built a RAG system over 100k documents and cut latency by 40%."


def test_ground_span_reslices_exact_substring():
    span = ground_span(
        document_id="doc-1",
        source_text=SOURCE,
        start_index=0,
        end_index=38,
    )
    assert span.text == "Built a RAG system over 100k documents"
    assert SOURCE[span.start_index : span.end_index] == span.text


def test_ground_span_captures_a_number_with_its_context():
    # the "40%" claim, grounded to the exact source characters
    start = SOURCE.index("cut latency by 40%")
    end = start + len("cut latency by 40%")
    span = ground_span(
        document_id="doc-1", source_text=SOURCE, start_index=start, end_index=end
    )
    assert span.text == "cut latency by 40%"


def test_out_of_bounds_indices_are_rejected():
    with pytest.raises(SpanGroundingError):
        ground_span(document_id="d", source_text=SOURCE, start_index=0, end_index=9999)


def test_inverted_indices_are_rejected():
    with pytest.raises(SpanGroundingError):
        ground_span(document_id="d", source_text=SOURCE, start_index=20, end_index=10)


def test_reverify_passes_when_source_unchanged():
    span = ground_span(document_id="d", source_text=SOURCE, start_index=0, end_index=38)
    assert reverify_span(span, SOURCE) is True


def test_reverify_fails_when_source_drifts():
    span = ground_span(document_id="d", source_text=SOURCE, start_index=0, end_index=38)
    drifted = "Prepended text. " + SOURCE  # indices now point at the wrong place
    assert reverify_span(span, drifted) is False


def test_span_text_length_must_match_indices():
    # constructing a SourceSpan whose text disagrees with its range must fail,
    # so a hand-built or model-built span can never lie about its own length
    from app.schemas.claim import SourceSpan

    with pytest.raises(ValueError):
        SourceSpan(document_id="d", start_index=0, end_index=5, text="way too long text")
