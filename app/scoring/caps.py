"""
The evidence cap (spec section 3): applied as an explicit, separately
testable post-processing step over the seven raw dimension scores, never
folded into any single dimension function.

> A profile with no proof cannot score well.
>
> When every claim on a profile is self-declared, the evidence dimension is
> capped at 3 out of 10, and the overall readiness score is capped at 30.

Kept as its own step (rather than, say, having score_evidence_quality clamp
itself) so the RAW dimension scores stay inspectable on their own terms - you
can always ask "what did evidence_quality actually compute before any cap?"
separately from "what does the user see after the cap?".
"""

from __future__ import annotations

from app.config.weights import EVIDENCE_DIMENSION_CAP, READINESS_CAP_WHEN_UNPROVEN
from app.schemas import Claim, DimensionScore, EvidenceTier


def all_claims_self_declared(claims: list[Claim]) -> bool:
    """
    True when there is no real proof anywhere: either every claim is T8
    (self-declared), or there are no claims at all - an empty profile has
    exactly as much proof as an all-self-declared one, which is none, so it
    triggers the same cap rather than vacuously escaping it.
    """
    return all(c.evidence_tier == EvidenceTier.T8 for c in claims)


def cap_reason(claims: list[Claim]) -> str:
    """
    Human-readable explanation of WHY the evidence cap fired, for display
    alongside `capped` (only meaningful when all_claims_self_declared(claims)
    is True). Distinguishes the two situations that boolean condition
    conflates:

      1. No claims at all - nothing has been extracted yet.
      2. Claims exist (possibly even grounded/publishable ones - a claim can
         be a verbatim, spot-checkable quote from the user's own CV/Upwork
         text and STILL be T8, since T8 means no THIRD-PARTY corroboration,
         not "ungrounded") but every single one tops out at T8.

    Getting this distinction right matters: "capped until claims are proven"
    is actively misleading when some claims already reverify against a real
    source span - the cap fired because nothing corroborates them, not
    because nothing points at a source. See CLAUDE.md's note on this exact
    conflation.
    """
    if not claims:
        return (
            "Capped at 30 — no evidence has been added yet. Add a CV, "
            "GitHub repo, or Upwork profile text to start building provable claims."
        )
    return (
        "Capped at 30 — every claim on this profile is self-declared "
        "(T8), with no third-party corroboration. Add a client review, a "
        "public repo, a platform assessment, or a verifiable certification "
        "to lift this."
    )


def apply_caps(
    dimensions: list[DimensionScore], claims: list[Claim]
) -> tuple[list[DimensionScore], float, bool]:
    """
    Given the raw, uncapped per-dimension scores, apply spec section 3's
    evidence cap when every claim is self-declared:
      - the evidence_quality dimension's displayed score is capped at
        EVIDENCE_DIMENSION_CAP (30/100)
      - the overall readiness (weighted sum of the - now possibly capped -
        dimension scores) is capped at READINESS_CAP_WHEN_UNPROVEN (30/100)

    Returns (dimensions with the cap applied where relevant, readiness,
    whether the cap fired). Dimension weights must sum to 1.0 (enforced by
    config.weights.validate_config at startup), so the weighted sum is
    already on a 0-100 scale with no extra normalization needed.
    """
    capped = all_claims_self_declared(claims)

    capped_dimensions: list[DimensionScore] = []
    for dimension in dimensions:
        if capped and dimension.name == "evidence_quality" and dimension.score > EVIDENCE_DIMENSION_CAP:
            capped_dimensions.append(
                dimension.model_copy(update={"score": EVIDENCE_DIMENSION_CAP})
            )
        else:
            capped_dimensions.append(dimension)

    readiness = sum(d.score * d.weight for d in capped_dimensions)
    if capped:
        readiness = min(readiness, READINESS_CAP_WHEN_UNPROVEN)

    return capped_dimensions, readiness, capped
