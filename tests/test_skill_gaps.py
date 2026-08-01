"""
Tests for find_skill_gaps (spec section 5): "which skills the top tier holds
that a user lacks... a learning signal, not a profile-writing fix."

The one invariant that matters most here isn't any single flagged/not-flagged
case - it's that this report is completely inert with respect to scoring.
See test_skill_gaps_do_not_affect_readiness_or_dimension_scores below.
"""

from __future__ import annotations

from app.schemas import Benchmark, Claim, EvidenceTier, RateBand, SourceSpan, SourceType
from app.scoring import score_profile
from app.scoring.skill_gaps import find_skill_gaps


def _benchmark(**overrides) -> Benchmark:
    defaults = dict(
        niche="Test Niche",
        version="2026-01",
        required_terms=["n8n", "webhook"],
        benchmark_topics=["workflow automation", "vector database"],
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
    skill_ids: list[str] | None = None,
    span_text: str | None = None,
    tier: EvidenceTier = EvidenceTier.T2,
    weight: float = 0.85,
) -> Claim:
    span = _span(span_text) if span_text is not None else None
    return Claim(
        claim_text=claim_text,
        skill_ids=skill_ids or [],
        source_type=SourceType.GITHUB_REPO,
        source_span=span,
        evidence_tier=tier,
        weight=weight,
    )


def test_topic_with_a_matching_claim_is_not_flagged():
    benchmark = _benchmark()
    claims = [
        _claim("built a workflow automation pipeline", span_text="built a workflow automation pipeline")
    ]

    gaps = find_skill_gaps(claims, benchmark)

    assert "workflow automation" not in gaps


def test_topic_with_zero_matching_claims_is_flagged():
    benchmark = _benchmark()
    claims = [
        _claim("built a workflow automation pipeline", span_text="built a workflow automation pipeline")
    ]

    gaps = find_skill_gaps(claims, benchmark)

    # Nothing anywhere in the evidence mentions vector databases/Pinecone/etc.
    assert "vector database" in gaps


def test_required_term_covered_only_via_skill_id_is_not_flagged():
    """
    A claim tagged skill_ids=['n8n'] by the extractor still counts as
    covering the "n8n" required term even when the claim/span text never
    literally says "n8n" - the normalized tag is closer to ground truth here
    than a string match would be.
    """
    benchmark = _benchmark(required_terms=["n8n"])
    claims = [
        _claim(
            "built an automation platform for a client",
            skill_ids=["n8n"],
            span_text="built an automation platform for a client",
        )
    ]

    gaps = find_skill_gaps(claims, benchmark)

    assert "n8n" not in gaps


def test_required_term_with_no_matching_claim_at_all_is_flagged():
    benchmark = _benchmark(required_terms=["webhook"])
    claims = [_claim("wrote some Python scripts", skill_ids=["python"], span_text="wrote some Python scripts")]

    gaps = find_skill_gaps(claims, benchmark)

    assert "webhook" in gaps


def test_a_term_present_in_both_lists_is_reported_only_once():
    benchmark = _benchmark(
        required_terms=["automation"],
        benchmark_topics=["automation"],
    )
    claims = [_claim("wrote some Python scripts", span_text="wrote some Python scripts")]

    gaps = find_skill_gaps(claims, benchmark)

    assert gaps.count("automation") == 1


def test_empty_claims_flags_every_required_term_and_topic():
    benchmark = _benchmark()
    gaps = find_skill_gaps([], benchmark)
    assert set(gaps) == {"n8n", "webhook", "workflow automation", "vector database"}


def test_find_skill_gaps_does_not_mutate_its_inputs():
    benchmark = _benchmark()
    claim = _claim("built a workflow automation pipeline", skill_ids=["n8n"], span_text="built a workflow automation pipeline")

    find_skill_gaps([claim], benchmark)

    assert claim.claim_text == "built a workflow automation pipeline"
    assert claim.skill_ids == ["n8n"]


def test_skill_gaps_do_not_affect_readiness_or_dimension_scores():
    """
    The load-bearing guarantee (spec section 5): skill-gap reporting is a
    SEPARATE, unscored output. score_profile must return byte-identical
    dimensions/readiness/capped whether or not find_skill_gaps is ever
    called alongside it, and "skill_gaps" must never appear as an eighth
    scoring dimension.
    """
    benchmark = _benchmark()
    claims = [
        _claim("built a workflow automation pipeline", skill_ids=["n8n"], span_text="built a workflow automation pipeline")
    ]

    dimensions_before, readiness_before, capped_before = score_profile(claims, benchmark)

    find_skill_gaps(claims, benchmark)  # the thing under test - must be inert

    dimensions_after, readiness_after, capped_after = score_profile(claims, benchmark)

    assert readiness_before == readiness_after
    assert capped_before == capped_after
    assert [d.model_dump() for d in dimensions_before] == [d.model_dump() for d in dimensions_after]
    assert "skill_gaps" not in {d.name for d in dimensions_before}
    assert len(dimensions_before) == 7
