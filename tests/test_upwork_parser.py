"""
Upwork parser tests. There is no API here - the pasted text IS the source
document, so claims ground directly against exactly what was pasted.
"""

from __future__ import annotations

import json

import pytest

from app.ingestion.upwork_parser import UpworkParsingError, parse_upwork_text
from app.schemas import SourceType
from app.storage.store import file_store

PASTED = (
    "5-star rated freelancer. Delivered a Zapier-to-Airtable automation that "
    "saved the client 10 hours per week."
)


class _FakeLLMClient:
    def __init__(self, body: str) -> None:
        self.body = body

    def complete(self, *, system: str, prompt: str) -> str:
        return self.body


def test_parse_upwork_text_grounds_against_pasted_text():
    body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "saved 10 hrs/week",
                    "skill_ids": ["zapier", "airtable"],
                    "evidence_quote": "saved the client 10 hours per week",
                }
            ]
        }
    )

    claims = parse_upwork_text(PASTED, client=_FakeLLMClient(body))

    published = [c for c in claims if c.publishable]
    assert published
    assert published[0].source_span.text == "saved the client 10 hours per week"
    assert published[0].source_type == SourceType.UPWORK_TEXT

    # The paste IS the source, but it's still stored as two linked documents:
    # the span's text document, and the retained original the text came from.
    text_id = published[0].source_span.document_id
    assert file_store.get_text(text_id) == PASTED
    original_id = file_store.get_original_id(text_id)
    assert file_store.get_bytes(original_id) == PASTED.encode("utf-8")


def test_parse_upwork_text_rejects_too_short_text():
    with pytest.raises(UpworkParsingError):
        parse_upwork_text("too short")


def test_parse_upwork_text_rejects_whitespace_only():
    with pytest.raises(UpworkParsingError):
        parse_upwork_text("      ")
