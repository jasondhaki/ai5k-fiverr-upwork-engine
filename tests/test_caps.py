"""
Tests for apply_caps (spec section 3's evidence cap), kept separately
testable from the raw dimension scores it caps.
"""

from __future__ import annotations

import pytest

from app.config.weights import EVIDENCE_DIMENSION_CAP, READINESS_CAP_WHEN_UNPROVEN
from app.schemas import Claim, DimensionScore, EvidenceTier, SourceSpan, SourceType
from app.scoring.caps import all_claims_self_declared, apply_caps, cap_reason


def _span(text: str) -> SourceSpan:
    return SourceSpan(document_id="d", start_index=0, end_index=len(text), text=text)


def _claim(tier: EvidenceTier) -> Claim:
    """apply_caps decides purely from evidence_tier, never from grounding
    status (that's a separate concern, already covered in test_tiers.py) -
    every claim here is grounded so tier is the only thing varying."""
    return Claim(
        claim_text="x",
        source_type=SourceType.CV,
        source_span=_span("x"),
        evidence_tier=tier,
        weight=0.5,
    )


def _dimensions(evidence_quality_score: float, **other_scores: float) -> list[DimensionScore]:
    scores = {
        "positioning": 50.0,
        "evidence_quality": evidence_quality_score,
        "keyword_coverage": 50.0,
        "portfolio_quality": 50.0,
        "completeness": 50.0,
        "conversion": 50.0,
        "pricing_strategy": 50.0,
        **other_scores,
    }
    weights = {
        "positioning": 0.22,
        "evidence_quality": 0.22,
        "keyword_coverage": 0.15,
        "portfolio_quality": 0.15,
        "completeness": 0.10,
        "conversion": 0.08,
        "pricing_strategy": 0.08,
    }
    return [DimensionScore(name=name, score=scores[name], weight=weights[name]) for name in weights]


# --- all_claims_self_declared -------------------------------------------------


def test_all_claims_self_declared_true_when_every_claim_is_t8():
    assert all_claims_self_declared([_claim(EvidenceTier.T8), _claim(EvidenceTier.T8)])


def test_all_claims_self_declared_false_when_any_claim_is_stronger():
    assert not all_claims_self_declared([_claim(EvidenceTier.T8), _claim(EvidenceTier.T2)])


def test_all_claims_self_declared_true_for_an_empty_list():
    """No proof at all is treated the same as all-self-declared proof - an
    empty profile doesn't vacuously escape the cap."""
    assert all_claims_self_declared([])


# --- apply_caps ----------------------------------------------------------------


def test_evidence_dimension_is_capped_when_all_claims_are_self_declared():
    dimensions = _dimensions(evidence_quality_score=90.0)
    claims = [_claim(EvidenceTier.T8), _claim(EvidenceTier.T8)]

    capped_dimensions, readiness, capped = apply_caps(dimensions, claims)

    evidence_dim = next(d for d in capped_dimensions if d.name == "evidence_quality")
    assert evidence_dim.score == pytest.approx(EVIDENCE_DIMENSION_CAP)
    assert capped is True


def test_readiness_is_capped_when_all_claims_are_self_declared():
    dimensions = _dimensions(evidence_quality_score=90.0, positioning=90.0, keyword_coverage=90.0)
    claims = [_claim(EvidenceTier.T8)]

    _, readiness, capped = apply_caps(dimensions, claims)

    assert readiness <= READINESS_CAP_WHEN_UNPROVEN
    assert capped is True


def test_caps_do_not_apply_when_real_evidence_exists():
    """The cap must not fire just because evidence_quality happens to be
    low - it fires only when EVERY claim is self-declared."""
    dimensions = _dimensions(evidence_quality_score=20.0)
    claims = [_claim(EvidenceTier.T2)]  # one real, non-self-declared claim

    capped_dimensions, readiness, capped = apply_caps(dimensions, claims)

    evidence_dim = next(d for d in capped_dimensions if d.name == "evidence_quality")
    assert evidence_dim.score == pytest.approx(20.0)  # unchanged
    assert capped is False


def test_evidence_score_below_the_cap_is_left_alone_even_when_capped():
    """The cap is a ceiling, not a floor or a forced value - a raw score
    already below it should not be pulled UP to the cap."""
    dimensions = _dimensions(evidence_quality_score=10.0)
    claims = [_claim(EvidenceTier.T8)]

    capped_dimensions, _, capped = apply_caps(dimensions, claims)

    evidence_dim = next(d for d in capped_dimensions if d.name == "evidence_quality")
    assert evidence_dim.score == pytest.approx(10.0)
    assert capped is True


def test_non_evidence_dimensions_are_never_capped():
    dimensions = _dimensions(evidence_quality_score=90.0, positioning=95.0)
    claims = [_claim(EvidenceTier.T8)]

    capped_dimensions, _, _ = apply_caps(dimensions, claims)

    positioning_dim = next(d for d in capped_dimensions if d.name == "positioning")
    assert positioning_dim.score == pytest.approx(95.0)


# --- cap_reason: the message must name the REAL condition that fired --------


def test_cap_reason_for_an_empty_profile_mentions_no_evidence():
    reason = cap_reason([])
    assert "no evidence" in reason.lower()
    assert "self-declared" not in reason.lower()


def test_cap_reason_for_self_declared_claims_mentions_self_declared_not_unproven():
    """The exact bug this guards against: a claim can be grounded (a real,
    verbatim quote) and STILL be T8 (self-declared - no third-party
    corroboration). The message must say THAT, not the misleading 'until
    claims are proven', which conflates groundedness with corroboration."""
    grounded_but_self_declared = Claim(
        claim_text="I am a great developer",
        source_type=SourceType.UPWORK_TEXT,
        source_span=_span("I am a great developer"),
        evidence_tier=EvidenceTier.T8,
        weight=0.15,
    )
    reason = cap_reason([grounded_but_self_declared])
    assert "self-declared" in reason.lower()
    assert "proven" not in reason.lower()


def test_cap_reason_distinguishes_the_two_situations():
    empty_reason = cap_reason([])
    self_declared_reason = cap_reason([_claim(EvidenceTier.T8)])
    assert empty_reason != self_declared_reason


def test_readiness_matches_the_weighted_sum_when_not_capped():
    dimensions = _dimensions(
        evidence_quality_score=50.0, positioning=50.0, keyword_coverage=50.0,
        portfolio_quality=50.0, completeness=50.0, conversion=50.0, pricing_strategy=50.0,
    )
    claims = [_claim(EvidenceTier.T2)]

    _, readiness, capped = apply_caps(dimensions, claims)

    assert capped is False
    assert readiness == pytest.approx(50.0)  # weights sum to 1.0, all scores equal
