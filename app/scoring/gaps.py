"""
rank_gaps: turns seven scored dimensions into a short, balanced action list
(spec section 6).

Four composable steps, each independently reasoned about:
  1. Score every gap - gain/priority, from the exact formula in the spec.
  2. Pull blocking items out into their own list - never part of the ranking.
  3. Apply dependency gating - a gap can be hidden until its prerequisite
     clears, even if its raw gain/priority would otherwise qualify.
  4. Assemble the balanced result: top three by priority, plus the single
     largest available gain regardless of effort (if not already included).

`blocking` and `gaps` stay two separate lists on the Result, matching the
schema (`Result.blocking` / `Result.gaps`) and the existing invariant that
blocking items are liabilities, not ranked opportunities - they are never
merged into the gap list itself, only presented alongside it.
"""

from __future__ import annotations

from app.config.weights import EFFICACY_TABLE, EFFORT_HOURS_FLOOR, EFFORT_HOURS_TABLE
from app.schemas import BlockingItem, BlockingReason, Claim, DimensionScore, Gap
from app.schemas.benchmark import Benchmark
from app.scoring.dimensions import _claim_text_pool, _contains


def _has_vertical_alignment(claims: list[Claim], benchmark: Benchmark) -> bool:
    """
    Reuses the exact same vocabulary-overlap check `score_positioning`
    itself computes (app/scoring/dimensions.py) - at least one claim's text
    touches this niche's own required_terms/benchmark_topics, not just ANY
    claim regardless of relevance. See the positioning gate's comment below
    for why this is the right tightening.
    """
    pool_lower = _claim_text_pool(claims).lower()
    niche_terms = [*benchmark.required_terms, *benchmark.benchmark_topics]
    return any(_contains(term, pool_lower) for term in niche_terms)


# Dimensions whose gap is hidden until a named prerequisite clears (spec
# section 6: "No title rewrite before a vertical is chosen. No pricing
# advice before evidence exists."). Every other dimension has no gate.
#
#   - positioning: spec's literal prerequisite - "a vertical is chosen" - is
#     trivially always true in THIS architecture, since the niche is picked
#     by the user up front (PipelineInput.niche is required) before any
#     ingestion even runs. Gating on bool(claims) alone (the original,
#     looser version) would let a single claim with ZERO relevance to the
#     niche unlock a "rewrite your title around this vertical" suggestion -
#     which is premature: there's nothing vertical-specific to reposition
#     around yet. Tightened to require at least one claim whose text
#     actually overlaps this niche's own vocabulary
#     (_has_vertical_alignment, reusing score_positioning's own signal
#     rather than inventing a second heuristic). With zero vertical-relevant
#     evidence, the right next step surfaces via evidence_quality/
#     keyword_coverage instead - both ungated - not a stylistic rewrite.
#
#   - pricing_strategy: "evidence exists" is already directly checkable from
#     available data - at least one claim that actually cleared grounding.
#     Deliberately NOT tightened further (e.g. to "at least one T1/T2-tier
#     claim"): that would conflate the GATE ("does any real evidence exist
#     at all") with the DIMENSION SCORE itself, which already measures "how
#     much of that evidence justifies the top rate band". A single grounded
#     T8 claim correctly clears the gate and then scores 0% on the metric -
#     two different questions, not one collapsed into the other.
_DEPENDENCY_GATES = {
    "positioning": _has_vertical_alignment,
    "pricing_strategy": lambda claims, benchmark: any(c.publishable for c in claims),
}


def _dependency_clear(dimension_name: str, claims: list[Claim], benchmark: Benchmark) -> bool:
    gate = _DEPENDENCY_GATES.get(dimension_name)
    return gate(claims, benchmark) if gate is not None else True


def _build_gap(dimension: DimensionScore, benchmark: Benchmark) -> Gap:
    """
    gain     = weight * (benchmark_target - current) * efficacy
    priority = gain / max(effort_hours, EFFORT_HOURS_FLOOR)

    `current` is the dimension's score AFTER apply_caps - the gap reflects
    what the user actually sees, not an uncapped number they never will.
    `target` is the niche's own benchmark value, never a perfect 100
    (spec section 6). gain is clipped at 0: a dimension already at or past
    its target has no gap to close, and Gap.gain is schema-constrained to
    be non-negative.
    """
    target = benchmark.dimension_targets[dimension.name]
    efficacy = EFFICACY_TABLE[dimension.name]
    effort_hours = max(EFFORT_HOURS_TABLE[dimension.name], EFFORT_HOURS_FLOOR)

    raw_gain = dimension.weight * (target - dimension.score) * efficacy
    gain = max(0.0, raw_gain)
    priority = gain / effort_hours

    return Gap(
        dimension=dimension.name,
        current=dimension.score,
        target=target,
        efficacy=efficacy,
        effort_hours=effort_hours,
        gain=gain,
        priority=priority,
    )


def _blocking_items(claims: list[Claim]) -> list[BlockingItem]:
    """
    Unproven claims are the one blocking signal derivable from current data.
    One aggregate item (not one per unproven claim) mirrors the walking
    skeleton's own example ("9 of 11 claims unproven").

    NOT YET IMPLEMENTED - READ BEFORE TRUSTING AN EMPTY BLOCKING LIST:
    BlockingReason.TOS_RISK and BlockingReason.MISSING_IDENTITY_VERIFICATION
    are NEVER raised anywhere in this function, or anywhere else in the
    codebase. This is not an oversight to route around - there is currently
    NO SIGNAL in the Claim/Benchmark schema that could tell this function a
    ToS risk exists (e.g. scraped-without-consent data) or that identity
    hasn't been verified. Both enum values stay reachable in BlockingReason
    for when that signal exists, but until then a real Result's `blocking`
    list reflects unproven claims ONLY. An empty (or unproven-claims-only)
    blocking list on a real report is NOT evidence that no ToS risk or
    identity-verification gap exists for that profile - it means this
    function has no way to know either way. Do not read "blocking is empty"
    as "nothing is blocking."
    """
    total = len(claims)
    unproven = sum(1 for c in claims if not c.publishable)
    if not unproven:
        return []
    return [
        BlockingItem(
            reason=BlockingReason.UNPROVEN_CLAIM,
            description=f"{unproven} of {total} claims unproven",
            dimension="evidence_quality",
        )
    ]


def rank_gaps(
    claims: list[Claim], dimensions: list[DimensionScore], benchmark: Benchmark
) -> tuple[list[Gap], list[BlockingItem]]:
    """
    Assemble the balanced gap list: candidates are every dimension whose
    dependency gate is clear and whose gain is strictly positive, ranked by
    priority (points per hour). The result is the top three by priority,
    plus the single largest-gain candidate if it isn't already among them -
    never padded out to a fixed count, and never including a gap that isn't
    worth making at all (zero gain) or isn't allowed yet (gated).
    """
    blocking = _blocking_items(claims)

    candidates = [
        _build_gap(dimension, benchmark)
        for dimension in dimensions
        if _dependency_clear(dimension.name, claims, benchmark)
    ]
    candidates = [gap for gap in candidates if gap.gain > 0]

    by_priority = sorted(candidates, key=lambda g: g.priority, reverse=True)
    top_three = by_priority[:3]

    largest_gain = max(candidates, key=lambda g: g.gain, default=None)
    included = {g.dimension for g in top_three}
    if largest_gain is not None and largest_gain.dimension not in included:
        gaps = [*top_three, largest_gain]
    else:
        gaps = top_three

    return gaps, blocking
