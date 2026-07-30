"""
Tests for the frozen schemas and the invariants they must never lose.
"""

from datetime import date

from app.config.weights import (
    DIMENSION_WEIGHTS,
    TIER_WEIGHTS,
    validate_config,
)
from app.schemas import Claim, SourceSpan, SourceType
from app.schemas.claim import EvidenceTier


def _span() -> SourceSpan:
    return SourceSpan(document_id="d", start_index=0, end_index=5, text="hello")


def test_claim_with_span_is_publishable():
    c = Claim(
        claim_text="x",
        source_type=SourceType.GITHUB_REPO,
        source_span=_span(),
        evidence_tier=EvidenceTier.T2,
        weight=0.85,
    )
    assert c.publishable is True


def test_claim_without_span_is_not_publishable():
    # THE GOVERNING RULE: no source span => never published.
    c = Claim(
        claim_text="I am great at everything",
        source_type=SourceType.CV,
        source_span=None,
        evidence_tier=EvidenceTier.T8,
        weight=0.15,
    )
    assert c.publishable is False


def test_effective_weight_discounts_by_recency():
    c = Claim(
        claim_text="old project",
        source_type=SourceType.CV,
        source_span=_span(),
        evidence_tier=EvidenceTier.T2,
        weight=0.85,
        observed_date=date(2019, 1, 1),
        recency_factor=0.5,
    )
    assert c.effective_weight == 0.425


def test_tier_weights_are_monotonic():
    # T1 strongest ... T8 weakest, strictly decreasing
    order = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    values = [TIER_WEIGHTS[t] for t in order]
    assert values == sorted(values, reverse=True)


def test_shipped_work_outranks_certification():
    # T2 (project demonstrated) must sit above T4 (proctored cert)
    assert TIER_WEIGHTS["T2"] > TIER_WEIGHTS["T4"]


def test_dimension_weights_sum_to_one():
    assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-6


def test_positioning_and_evidence_are_44_percent():
    assert abs((DIMENSION_WEIGHTS["positioning"] + DIMENSION_WEIGHTS["evidence_quality"]) - 0.44) < 1e-6


def test_config_validation_passes():
    validate_config()  # raises if inconsistent
