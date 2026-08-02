"""
Proves the walking skeleton runs the whole path and returns a valid Result.
This must stay green as every stub is replaced by a real component.
"""

import json

import pytest

from app.platform.pipeline import PipelineInput, run_pipeline
from app.schemas import Result


class _FakeLLMClient:
    """No network, no API key: the extractor's LLM boundary is the only
    thing faked here, so this still runs the real router/parser/grounding path."""

    def complete(self, *, system: str, prompt: str) -> str:
        return '{"claims": []}'


@pytest.fixture(autouse=True)
def _no_live_generation_calls(monkeypatch):
    """
    Every test below runs with empty (or ungrounded-only) claim lists, so
    generate_title_draft/generate_overview_draft return None before ever
    calling .complete() (see app/generation/generator.py's early-return
    guards) - no live call would happen even without this. Monkeypatched
    anyway so that safety is explicit and doesn't depend on that invariant
    holding as this file's fixtures evolve, matching the same pattern
    already used for ingestion's build_default_client below.
    """
    monkeypatch.setattr("app.generation.build_default_client", lambda: _FakeLLMClient())


def test_pipeline_returns_valid_result(monkeypatch):
    monkeypatch.setattr(
        "app.ingestion.upwork_parser.build_default_client", lambda: _FakeLLMClient()
    )
    inp = PipelineInput(niche="SMB workflow automation", upwork_text="I am an AI developer")
    result = run_pipeline(inp)
    assert isinstance(result, Result)


def test_result_has_all_seven_dimensions():
    result = run_pipeline(PipelineInput(niche="SMB workflow automation"))
    assert len(result.dimensions) == 7


def test_blocking_is_separate_from_gaps():
    result = run_pipeline(PipelineInput(niche="SMB workflow automation"))
    # blocking items are their own list, never mixed into ranked gaps
    assert isinstance(result.blocking, list)
    assert isinstance(result.gaps, list)


def test_provable_count_never_exceeds_total():
    result = run_pipeline(PipelineInput(niche="SMB workflow automation"))
    assert result.provable_claims <= result.total_claims


def test_scoring_is_real_not_the_old_stub_data():
    """
    score_profile/rank_gaps used to return fixed fake data regardless of
    input (readiness always 41.0, evidence_quality detail always literally
    "2 of 11 claims proven"). With zero claims and the real implementation,
    the evidence cap must fire (no proof at all is treated the same as
    all-self-declared) and the detail must reflect the ACTUAL claim count,
    not the old stub's fake one.
    """
    result = run_pipeline(PipelineInput(niche="SMB workflow automation"))

    assert result.total_claims == 0
    assert result.capped is True
    assert result.readiness <= 30.0

    evidence_dim = next(d for d in result.dimensions if d.name == "evidence_quality")
    assert evidence_dim.score <= 30.0
    assert evidence_dim.detail == "No claims yet"  # not the old stub's fake "2 of 11"


def test_benchmark_version_on_result_matches_the_loaded_benchmark():
    result = run_pipeline(PipelineInput(niche="SMB workflow automation"))
    assert result.benchmark_version == "2026-07"


def test_skill_gaps_populated_end_to_end_with_zero_claims():
    """With no evidence at all, every required term/topic in the real
    SMB-automation benchmark is a gap - and it must be a plain list on
    Result, never affecting the (already-asserted-capped) readiness."""
    result = run_pipeline(PipelineInput(niche="SMB workflow automation"))
    assert result.skill_gaps  # non-empty - nothing covers anything with 0 claims
    assert "n8n" in result.skill_gaps
    assert isinstance(result.skill_gaps, list)


def test_generation_incomplete_is_surfaced_on_the_result_when_retries_exhaust(monkeypatch):
    """
    End-to-end proof that a real report can distinguish "generation kept
    failing validation" from "nothing to generate yet" via
    Result.generation_incomplete - not only visible in a log line. Without
    this, a user seeing no generated title has no way to tell the two apart.
    """
    ingestion_body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "cut costs by 40 percent",
                    "skill_ids": ["automation"],
                    "evidence_quote": "cut costs by 40 percent for a client",
                }
            ]
        }
    )

    class _FakeIngestClient:
        def complete(self, *, system: str, prompt: str) -> str:
            return ingestion_body

    class _AlwaysFabricatesGenClient:
        def complete(self, *, system: str, prompt: str) -> str:
            # Cites a number no real claim backs - fails validation every time.
            return json.dumps({"text": "Achieved a 999 percent improvement"})

    monkeypatch.setattr(
        "app.ingestion.upwork_parser.build_default_client", lambda: _FakeIngestClient()
    )
    monkeypatch.setattr(
        "app.generation.build_default_client", lambda: _AlwaysFabricatesGenClient()
    )

    inp = PipelineInput(
        niche="SMB workflow automation",
        upwork_text="cut costs by 40 percent for a client this year",
    )
    result = run_pipeline(inp)

    assert result.total_claims == 1  # real evidence existed - this isn't the empty case
    assert result.generation_incomplete is True


# --- cap_note: names the real reason the cap fired, not a generic one -------
#
# Spec section 3's cap fires purely from evidence_tier (all T8), never from
# groundedness - a claim can be a real, verbatim quote and still be T8 (no
# THIRD-PARTY corroboration). The old static message ("Capped at 30 until
# claims are proven") conflated the two; cap_note must now say which
# situation actually produced the cap.


def test_cap_note_mentions_no_evidence_when_there_are_no_claims_at_all():
    result = run_pipeline(PipelineInput(niche="SMB workflow automation"))
    assert result.capped is True
    assert "no evidence" in result.cap_note.lower()
    assert "self-declared" not in result.cap_note.lower()


def test_cap_note_mentions_self_declared_when_claims_exist_but_are_all_t8(monkeypatch):
    ingestion_body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "I am a hard worker",
                    "skill_ids": [],
                    "evidence_quote": "I am a hard worker who never gives up",
                }
            ]
        }
    )

    class _FakeIngestClient:
        def complete(self, *, system: str, prompt: str) -> str:
            return ingestion_body

    monkeypatch.setattr(
        "app.ingestion.upwork_parser.build_default_client", lambda: _FakeIngestClient()
    )

    inp = PipelineInput(
        niche="SMB workflow automation",
        upwork_text="I am a hard worker who never gives up and loves my job",
    )
    result = run_pipeline(inp)

    assert result.total_claims == 1  # real, grounded claims - not the empty case
    assert result.capped is True
    assert "self-declared" in result.cap_note.lower()
    assert "until claims are proven" not in result.cap_note.lower()


# --- overview_blocked_by_evidence_tier: explain absent output, don't hide it -


def test_overview_blocked_by_evidence_tier_when_no_t1_through_t4_claims_exist(monkeypatch):
    """No T1-T4 evidence anywhere - the overview is never even attempted, and
    the result must say so explicitly rather than leaving an unexplained gap
    that generation_incomplete can't describe (nothing was tried at all)."""
    ingestion_body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "I am a hard worker",
                    "skill_ids": [],
                    "evidence_quote": "I am a hard worker who never gives up",
                }
            ]
        }
    )

    class _FakeIngestClient:
        def complete(self, *, system: str, prompt: str) -> str:
            return ingestion_body

    monkeypatch.setattr(
        "app.ingestion.upwork_parser.build_default_client", lambda: _FakeIngestClient()
    )

    inp = PipelineInput(
        niche="SMB workflow automation",
        upwork_text="I am a hard worker who never gives up and loves my job",
    )
    result = run_pipeline(inp)

    assert result.total_claims == 1
    assert result.overview_blocked_by_evidence_tier is True
    assert result.generation_incomplete is False  # never attempted, not a failure


def test_overview_not_blocked_when_t1_through_t4_evidence_exists(monkeypatch):
    ingestion_body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "5 star review noting saved time",
                    "skill_ids": [],
                    "evidence_quote": "5-star review: client said the automation saved 10 hours a week",
                }
            ]
        }
    )

    class _FakeIngestClient:
        def complete(self, *, system: str, prompt: str) -> str:
            return ingestion_body

    monkeypatch.setattr(
        "app.ingestion.upwork_parser.build_default_client", lambda: _FakeIngestClient()
    )

    inp = PipelineInput(
        niche="SMB workflow automation",
        upwork_text="5-star review: client said the automation saved 10 hours a week for their team",
    )
    result = run_pipeline(inp)

    assert result.overview_blocked_by_evidence_tier is False


def test_overview_blocked_by_evidence_tier_is_false_with_zero_claims_and_overview_present():
    """Sanity check on the flag's definition: it's False whenever an
    overview actually made it into `generated`, regardless of anything else -
    the two are mutually exclusive by construction (see run_pipeline)."""
    result = run_pipeline(PipelineInput(niche="SMB workflow automation"))
    overview_present = any(a.kind == "overview" for a in result.generated)
    assert overview_present is False  # zero claims - nothing was generated
    # and with nothing generated, it degrades to "blocked" - the zero-claims
    # case is a special case of "no T1-T4 evidence", not an exception to it
    assert result.overview_blocked_by_evidence_tier is True
