"""
Tests for rank_gaps (spec section 6): the gain/priority formula exactly as
specified, blocking items pulled into their own list, dependency gating, and
the balanced top-three-plus-largest-gain assembly.
"""

from __future__ import annotations

import pytest

from app.config.weights import DIMENSION_WEIGHTS, EFFICACY_TABLE, EFFORT_HOURS_FLOOR, EFFORT_HOURS_TABLE
from app.schemas import (
    Benchmark,
    BlockingReason,
    Claim,
    DimensionScore,
    EvidenceTier,
    Gap,
    RateBand,
    SourceSpan,
    SourceType,
)
from app.scoring.gaps import rank_gaps


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


def _dimension(name: str, score: float) -> DimensionScore:
    return DimensionScore(name=name, score=score, weight=DIMENSION_WEIGHTS[name])


def _claim(grounded: bool = True, text: str = "x") -> Claim:
    span = SourceSpan(document_id="d", start_index=0, end_index=len(text), text=text) if grounded else None
    return Claim(
        claim_text=text,
        source_type=SourceType.CV,
        source_span=span,
        evidence_tier=EvidenceTier.T2 if grounded else EvidenceTier.T8,
        weight=0.85 if grounded else 0.15,
    )


def _vertical_relevant_claim(grounded: bool = True) -> Claim:
    """A claim whose text actually overlaps this file's _benchmark()
    required_terms (["n8n"]) - clears the tightened positioning gate,
    unlike a generic claim with no relation to the niche at all."""
    return _claim(grounded=grounded, text="Built an n8n workflow")


# --- gain/priority formula -----------------------------------------------------


def test_gain_and_priority_match_the_spec_formula_exactly():
    benchmark = _benchmark()
    dimension = _dimension("keyword_coverage", score=40.0)
    claims = [_claim(grounded=True)]  # clears every dependency gate

    gaps, _ = rank_gaps(claims, [dimension], benchmark)

    target = benchmark.dimension_targets["keyword_coverage"]
    weight = DIMENSION_WEIGHTS["keyword_coverage"]
    efficacy = EFFICACY_TABLE["keyword_coverage"]
    effort = max(EFFORT_HOURS_TABLE["keyword_coverage"], EFFORT_HOURS_FLOOR)
    expected_gain = weight * (target - 40.0) * efficacy
    expected_priority = expected_gain / effort

    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.current == pytest.approx(40.0)
    assert gap.target == pytest.approx(target)
    assert gap.efficacy == pytest.approx(efficacy)
    assert gap.effort_hours == pytest.approx(effort)
    assert gap.gain == pytest.approx(expected_gain)
    assert gap.priority == pytest.approx(expected_priority)


def test_gain_is_clipped_at_zero_and_the_gap_is_dropped_when_target_is_already_met():
    benchmark = _benchmark()
    dimension = _dimension("keyword_coverage", score=100.0)  # already past the target of 90
    claims = [_claim(grounded=True)]

    gaps, _ = rank_gaps(claims, [dimension], benchmark)

    assert gaps == []  # no gain, no gap - never a negative-gain entry either


def test_dimensions_at_or_above_target_never_backfill_an_under_populated_list():
    """
    Zero-gain dimensions are filtered out of `candidates` BEFORE the top-
    three/largest-gain assembly even runs (see rank_gaps), so they can never
    be pulled in as filler just to pad the list toward five - even when
    genuine gaps are scarce. Covers both "exactly meets" (positioning,
    conversion) and "exceeds" (completeness, pricing_strategy) target cases
    in one scenario, with only ONE dimension left with a real gap to close.
    """
    benchmark = _benchmark()
    dimensions = [
        _dimension("keyword_coverage", score=40.0),  # target 90 - the one real gap
        _dimension("positioning", score=85.0),  # exactly at its target (85)
        _dimension("completeness", score=95.0),  # above its target (85)
        _dimension("conversion", score=80.0),  # exactly at its target (80)
        _dimension("pricing_strategy", score=100.0),  # above its target (80)
    ]
    claims = [_vertical_relevant_claim(grounded=True)]  # clears positioning's and pricing's gates

    gaps, _ = rank_gaps(claims, dimensions, benchmark)

    assert len(gaps) == 1
    assert gaps[0].dimension == "keyword_coverage"


# --- dependency gating -----------------------------------------------------------


def test_positioning_gap_is_hidden_with_zero_claims():
    benchmark = _benchmark()
    dimension = _dimension("positioning", score=30.0)

    gaps, _ = rank_gaps([], [dimension], benchmark)

    assert gaps == []


def test_positioning_gap_is_hidden_when_claims_exist_but_none_are_vertical_relevant():
    """The tightened gate: bare claim presence isn't enough - spec's literal
    prerequisite ("a vertical is chosen") is trivially always true here
    since the niche is picked before ingestion even runs, so the real
    question is whether any evidence actually touches that niche's
    vocabulary. A claim with zero relation to the niche must not unlock a
    'rewrite your title around this vertical' suggestion."""
    benchmark = _benchmark()
    dimension = _dimension("positioning", score=30.0)
    claims = [_claim(grounded=False, text="I am a hard worker")]  # present, but not vertical-relevant

    gaps, _ = rank_gaps(claims, [dimension], benchmark)

    assert gaps == []


def test_positioning_gap_appears_once_a_vertical_relevant_claim_exists():
    benchmark = _benchmark()
    dimension = _dimension("positioning", score=30.0)
    claims = [_vertical_relevant_claim(grounded=False)]  # doesn't even need to be proven - just relevant

    gaps, _ = rank_gaps(claims, [dimension], benchmark)

    assert len(gaps) == 1
    assert gaps[0].dimension == "positioning"


def test_pricing_strategy_gap_is_hidden_without_any_grounded_evidence():
    benchmark = _benchmark()
    dimension = _dimension("pricing_strategy", score=30.0)
    claims = [_claim(grounded=False)]  # claims exist, but none are proven

    gaps, _ = rank_gaps(claims, [dimension], benchmark)

    assert gaps == []


def test_pricing_strategy_gap_appears_once_grounded_evidence_exists():
    benchmark = _benchmark()
    dimension = _dimension("pricing_strategy", score=30.0)
    claims = [_claim(grounded=True)]

    gaps, _ = rank_gaps(claims, [dimension], benchmark)

    assert len(gaps) == 1
    assert gaps[0].dimension == "pricing_strategy"


def test_ungated_dimensions_have_no_prerequisite():
    benchmark = _benchmark()
    dimension = _dimension("keyword_coverage", score=30.0)

    gaps, _ = rank_gaps([], [dimension], benchmark)  # zero claims at all

    assert len(gaps) == 1
    assert gaps[0].dimension == "keyword_coverage"


# --- blocking items --------------------------------------------------------------


def test_unproven_claims_produce_one_aggregate_blocking_item():
    claims = [_claim(grounded=True), _claim(grounded=False), _claim(grounded=False)]

    _, blocking = rank_gaps(claims, [], _benchmark())

    assert len(blocking) == 1
    assert blocking[0].reason == BlockingReason.UNPROVEN_CLAIM
    assert blocking[0].description == "2 of 3 claims unproven"
    assert blocking[0].dimension == "evidence_quality"


def test_no_blocking_items_when_every_claim_is_proven():
    claims = [_claim(grounded=True), _claim(grounded=True)]

    _, blocking = rank_gaps(claims, [], _benchmark())

    assert blocking == []


def test_blocking_items_never_appear_in_the_gaps_list():
    claims = [_claim(grounded=False)]
    dimension = _dimension("keyword_coverage", score=30.0)

    gaps, blocking = rank_gaps(claims, [dimension], _benchmark())

    assert blocking  # the unproven claim did produce a blocking item
    assert all(isinstance(g, Gap) for g in gaps)
    assert len(gaps) == 1  # keyword_coverage's own gap, unaffected by blocking


# --- the balanced top-three-plus-largest-gain assembly ----------------------------


def test_top_three_by_priority_plus_the_largest_gain_are_assembled():
    """
    With every dimension's current score at 0 against this benchmark's real
    targets, and the REAL config weights/efficacy/effort-hours: the top
    three by priority are completeness, keyword_coverage, and
    pricing_strategy - but the single largest gain overall belongs to
    positioning, which ISN'T among those three. It must still be added as a
    fourth gap, proving the "largest gain regardless of effort" rule
    actually fires rather than being redundant with the top-three cut.
    """
    benchmark = _benchmark()
    dimensions = [_dimension(name, score=0.0) for name in DIMENSION_WEIGHTS]
    claims = [_vertical_relevant_claim(grounded=True)]  # clears both dependency gates

    gaps, _ = rank_gaps(claims, dimensions, benchmark)

    assert {g.dimension for g in gaps} == {
        "completeness", "keyword_coverage", "pricing_strategy", "positioning",
    }
    assert len(gaps) == 4


def test_largest_gain_is_not_duplicated_when_already_among_the_top_three():
    benchmark = _benchmark()
    # Only the three dimensions that were already the "top three by
    # priority" above - keyword_coverage also happens to have the largest
    # gain among just these three, so nothing extra should be appended.
    dimensions = [
        _dimension("completeness", score=0.0),
        _dimension("keyword_coverage", score=0.0),
        _dimension("pricing_strategy", score=0.0),
    ]
    claims = [_claim(grounded=True)]

    gaps, _ = rank_gaps(claims, dimensions, benchmark)

    assert len(gaps) == 3
    assert {g.dimension for g in gaps} == {"completeness", "keyword_coverage", "pricing_strategy"}


def test_gaps_are_ordered_by_priority_descending():
    benchmark = _benchmark()
    dimensions = [_dimension(name, score=0.0) for name in DIMENSION_WEIGHTS]
    claims = [_vertical_relevant_claim(grounded=True)]

    gaps, _ = rank_gaps(claims, dimensions, benchmark)

    priorities = [g.priority for g in gaps[:3]]
    assert priorities == sorted(priorities, reverse=True)
