"""
score_profile: runs the seven pure dimension functions, wraps each result as
a DimensionScore (with its config weight and a short human-readable detail),
then applies the evidence cap as an explicit final step.
"""

from __future__ import annotations

from app.config.weights import DIMENSION_WEIGHTS
from app.schemas import Benchmark, Claim, DimensionScore, SourceType
from app.scoring.caps import apply_caps
from app.scoring.dimensions import (
    _NUMBER_PATTERN,
    _OUTCOME_FRAMING_PATTERN,
    _PORTFOLIO_SOURCES,
    _ROLE_TITLE_PATTERN,
    _claim_text_pool,
    _contains,
    score_completeness,
    score_conversion,
    score_evidence_quality,
    score_keyword_coverage,
    score_portfolio_quality,
    score_positioning,
    score_pricing_strategy,
)

# Order matters only for presentation; scoring itself has no ordering
# dependency between dimensions.
_DIMENSION_FUNCTIONS = {
    "positioning": score_positioning,
    "evidence_quality": score_evidence_quality,
    "keyword_coverage": score_keyword_coverage,
    "portfolio_quality": score_portfolio_quality,
    "completeness": score_completeness,
    "conversion": score_conversion,
    "pricing_strategy": score_pricing_strategy,
}


def _detail(name: str, claims: list[Claim], benchmark: Benchmark) -> str | None:
    """A short 'you now' summary per dimension - deliberately simple, plain
    counts rather than restating the score, matching the walking skeleton's
    own example details (e.g. '2 of 11 claims proven')."""
    if name == "positioning":
        pool = _claim_text_pool(claims)
        pool_lower = pool.lower()
        niche_terms = [*benchmark.required_terms, *benchmark.benchmark_topics]
        signals = [
            bool(_ROLE_TITLE_PATTERN.search(pool)),
            any(_contains(term, pool_lower) for term in niche_terms),
            bool(_NUMBER_PATTERN.search(pool)),
        ]
        return f"{sum(signals)} of 3 positioning signals present"

    if name == "evidence_quality":
        total = len(claims)
        provable = sum(1 for c in claims if c.publishable)
        return f"{provable} of {total} claims proven" if total else "No claims yet"

    if name == "keyword_coverage":
        pool_lower = _claim_text_pool(claims).lower()
        if not benchmark.required_terms:
            return "No required terms in this benchmark"
        present = sum(1 for term in benchmark.required_terms if _contains(term, pool_lower))
        missing = len(benchmark.required_terms) - present
        return "All required terms present" if missing == 0 else f"Missing {missing} required terms"

    if name == "portfolio_quality":
        portfolio_claims = [c for c in claims if c.source_type in _PORTFOLIO_SOURCES and c.publishable]
        item_ids = {c.source_span.document_id for c in portfolio_claims}
        quantified_ids = {
            c.source_span.document_id
            for c in portfolio_claims
            if _NUMBER_PATTERN.search(c.claim_text) or _NUMBER_PATTERN.search(c.source_span.text)
        }
        return f"{len(item_ids)} item(s), {len(quantified_ids)} quantified"

    if name == "completeness":
        # Mirrors score_completeness's own four-item checklist.
        has_identity_history = any(c.source_type in (SourceType.CV, SourceType.LINKEDIN_EXPORT) for c in claims)
        has_portfolio_evidence = any(c.source_type in _PORTFOLIO_SOURCES for c in claims)
        distinct_skills = {skill for c in claims for skill in c.skill_ids}
        has_skill_breadth = len(distinct_skills) >= 3
        has_client_facing_text = any(
            c.source_type in (SourceType.UPWORK_TEXT, SourceType.ONBOARDING_FORM) for c in claims
        )
        present = sum([has_identity_history, has_portfolio_evidence, has_skill_breadth, has_client_facing_text])
        return f"{present} of 4 checklist items present"

    if name == "conversion":
        total = len(claims)
        if not total:
            return "No claims yet"
        outcome_framed = sum(
            1 for c in claims if _OUTCOME_FRAMING_PATTERN.search(c.claim_text) or _NUMBER_PATTERN.search(c.claim_text)
        )
        return f"{outcome_framed} of {total} claims outcome-framed"

    if name == "pricing_strategy":
        grounded = [c for c in claims if c.publishable]
        if not grounded:
            return "No proven claims yet"
        justifying_tiers = set(benchmark.rate_band.justifying_tiers)
        matching = sum(1 for c in grounded if c.evidence_tier.value in justifying_tiers)
        return f"{matching} of {len(grounded)} proven claims support the top rate band"

    return None  # pragma: no cover - every known dimension name is handled above


def score_profile(claims: list[Claim], benchmark: Benchmark) -> tuple[list[DimensionScore], float, bool]:
    """
    Compute all seven dimensions against `benchmark`, wrap them as
    DimensionScores, then apply the evidence cap (spec section 3) as an
    explicit final step. Returns (dimensions, readiness, capped) - the exact
    shape `run_pipeline` has always expected.
    """
    raw_dimensions = [
        DimensionScore(
            name=name,
            score=score_fn(claims, benchmark),
            weight=DIMENSION_WEIGHTS[name],
            detail=_detail(name, claims, benchmark),
        )
        for name, score_fn in _DIMENSION_FUNCTIONS.items()
    ]
    return apply_caps(raw_dimensions, claims)
