"""
Skill-gap reporting (spec section 5): "which skills the top tier holds that a
user lacks... a learning signal, not a profile-writing fix."

This is deliberately NOT a scoring dimension. score_profile/rank_gaps never
import or call anything here, and Result.skill_gaps is never read by
apply_caps or the readiness computation - penalising someone for a skill they
haven't built yet would be both unfair and useless (spec's own words). It is
purely a separate report, computed alongside scoring and attached to the same
Result for convenience, nothing more.
"""

from __future__ import annotations

from app.schemas import Benchmark, Claim
from app.scoring.dimensions import _claim_text_pool, _contains, _topic_covered


def find_skill_gaps(claims: list[Claim], benchmark: Benchmark) -> list[str]:
    """
    Every benchmark required_term / benchmark_topic with NO matching claim
    anywhere in the evidence - not "scored low," just absent entirely.

    A term/topic counts as covered if either:
      - it appears in some claim's normalized skill_ids (the closest thing to
        ground truth for "this claim demonstrates X" - more reliable than
        string-matching for a term the extractor tagged from context but
        didn't repeat verbatim), or
      - it's present in the claim-text/span vocabulary pool, via the exact
        same _contains/_topic_covered checks score_keyword_coverage already
        uses for "is X covered" - reused rather than reimplemented, so this
        report can never quietly disagree with the keyword-coverage
        dimension about what counts as present.

    Order matches benchmark.required_terms followed by benchmark.
    benchmark_topics; a term/topic present in both lists is only ever
    reported once, under whichever list it's missing from first.
    """
    pool_lower = _claim_text_pool(claims).lower()
    skill_ids_lower = {skill_id.lower() for claim in claims for skill_id in claim.skill_ids}

    gaps: list[str] = []
    seen: set[str] = set()

    for term in benchmark.required_terms:
        if term in seen:
            continue
        seen.add(term)
        if term.lower() in skill_ids_lower or _contains(term, pool_lower):
            continue
        gaps.append(term)

    for topic in benchmark.benchmark_topics:
        if topic in seen:
            continue
        seen.add(topic)
        if topic.lower() in skill_ids_lower or _topic_covered(topic, pool_lower):
            continue
        gaps.append(topic)

    return gaps
