"""
Tests for score_profile's per-dimension `detail` strings (app/scoring/profile.py) -
kept separate from test_dimensions.py, which only covers the pure score
functions themselves, not the human-readable summaries wrapped around them.
"""

from __future__ import annotations

from app.schemas import Benchmark, Claim, EvidenceTier, RateBand, SourceSpan, SourceType
from app.scoring.profile import score_profile


def _benchmark(**overrides) -> Benchmark:
    defaults = dict(
        niche="Test Niche",
        version="2026-01",
        required_terms=["n8n"],
        benchmark_topics=[],
        title_formula="[role] - [vertical] - [outcome]",
        overview_words_min=280,
        overview_words_max=420,
        portfolio_min_items=3,
        portfolio_min_quantified=2,
        rate_band=RateBand(low=50.0, high=80.0, justifying_tiers=["T1", "T2"]),
        dimension_targets={
            "positioning": 85.0,
            "evidence_quality": 80.0,
            "keyword_coverage": 90.0,
            "portfolio_quality": 85.0,
            "completeness": 85.0,
            "conversion": 80.0,
            "pricing_strategy": 80.0,
        },
    )
    defaults.update(overrides)
    return Benchmark(**defaults)


def _span(text: str, document_id: str = "doc-1") -> SourceSpan:
    return SourceSpan(document_id=document_id, start_index=0, end_index=len(text), text=text)


def _claim(
    claim_text: str,
    source_type: SourceType,
    span_text: str | None = None,
    document_id: str = "doc-1",
    tier: EvidenceTier = EvidenceTier.T2,
    weight: float = 0.85,
) -> Claim:
    span = _span(span_text, document_id) if span_text is not None else None
    return Claim(
        claim_text=claim_text,
        source_type=source_type,
        source_span=span,
        evidence_tier=tier,
        weight=weight,
    )


def _portfolio_detail(claims: list[Claim], benchmark: Benchmark | None = None) -> str:
    dimensions, _readiness, _capped = score_profile(claims, benchmark or _benchmark())
    return next(d for d in dimensions if d.name == "portfolio_quality").detail


# --- portfolio_quality: a zero must say WHY, not just show a bare count -----


def test_portfolio_detail_says_no_sources_provided_when_none_exist_at_all():
    """No GitHub/portfolio/demo claim anywhere - the score is 0 because
    nothing was ever supplied, not because supplied evidence failed."""
    non_portfolio_claim = _claim("cv text", SourceType.CV, span_text="cv text")
    detail = _portfolio_detail([non_portfolio_claim])
    assert "no portfolio sources" in detail.lower()
    assert "0 item" not in detail


def test_portfolio_detail_says_none_qualified_when_sources_exist_but_ungrounded():
    """A portfolio source WAS provided (a GitHub repo claim), but it never
    grounded - this must read differently from 'nothing was ever supplied'."""
    ungrounded_repo_claim = _claim("claims a repo", SourceType.GITHUB_REPO, span_text=None)
    detail = _portfolio_detail([ungrounded_repo_claim])
    assert "none produced a provable item" in detail.lower()
    assert "no portfolio sources" not in detail.lower()


def test_portfolio_detail_shows_real_counts_when_items_exist():
    claim = _claim("shipped project", SourceType.GITHUB_REPO, span_text="shipped project cut costs 40%")
    detail = _portfolio_detail([claim])
    assert "1 item" in detail


# --- DimensionScore.provisional: set from is_provisional, not a second list -


def test_score_profile_marks_the_four_heuristic_dimensions_provisional():
    dimensions, _readiness, _capped = score_profile([], _benchmark())
    by_name = {d.name: d.provisional for d in dimensions}

    assert by_name["positioning"] is True
    assert by_name["completeness"] is True
    assert by_name["conversion"] is True
    assert by_name["pricing_strategy"] is True
    assert by_name["evidence_quality"] is False
    assert by_name["keyword_coverage"] is False
    assert by_name["portfolio_quality"] is False


def test_portfolio_detail_three_states_are_all_distinct():
    no_sources = _portfolio_detail([_claim("cv text", SourceType.CV, span_text="cv text")])
    none_qualified = _portfolio_detail(
        [_claim("claims a repo", SourceType.GITHUB_REPO, span_text=None)]
    )
    has_items = _portfolio_detail(
        [_claim("shipped project", SourceType.GITHUB_REPO, span_text="shipped project")]
    )
    assert len({no_sources, none_qualified, has_items}) == 3
