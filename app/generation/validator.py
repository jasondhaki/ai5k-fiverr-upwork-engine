"""
The span validator: the ONLY place in the codebase that constructs a
GeneratedAsset with validated=True.

Built and tested TOGETHER with the generator (app/generation/generator.py)
in the same change, per CLAUDE.md. Structural guarantee: generator.py works
entirely in terms of DraftAsset, a type with no `validated` field at all -
there is no path from an LLM response to Result.generated that doesn't pass
through validate_asset() here, because nothing but this function ever
produces the type Result.generated actually holds.

This is the same defence span_grounding.py already established for claims,
applied one layer up: a model fabricating a claim and grounding it to an
unrelated real quote is exactly the failure mode this guards against for
GENERATED TEXT - a model writing a plausible-sounding number that happens to
be near, but not present in, the real evidence.
"""

from __future__ import annotations

import re

from app.generation.generator import PROOF_TIERS, DraftAsset
from app.ingestion.span_grounding import reverify_span
from app.schemas import GeneratedAsset, SourceSpan
from app.storage.store import file_store

# --- Number matching: the exact rule, spelled out ----------------------------
#
# A bare digit-substring check (`"20" in "2019 to 2024"`) is a real bypass
# vector: "20" IS a substring of "2019" purely by coincidence, so a
# fabricated "20%" would previously validate against ANY span that happened
# to mention an unrelated year, id number, or any other digit run containing
# "20" somewhere inside it. Two independent hardenings fix this:
#
#   1. WORD BOUNDARIES around the number core. `\b20\b` does NOT match
#      inside "2019" (there's no boundary between "0" and the following "1" -
#      both are \w characters), but DOES match a genuinely standalone "20".
#      The boundary is placed around the DIGIT CORE only, never including a
#      trailing unit like "%" - "%" is itself a non-word character, so a
#      literal `\b` placed right after it would fail to match at all when
#      followed by whitespace or end-of-string (both sides would be \W, and
#      \b never matches there). Unit context is therefore checked
#      separately, immediately following the core number, not baked into
#      the same boundary.
#   2. UNIT/CONTEXT AGREEMENT. If the generated text presents a number as a
#      percentage ("40%" / "40 percent"), the backing claim's span must ALSO
#      present that same number as a percentage nearby - not just contain
#      the bare digits in some unrelated context (a count, an id, a year).
#      This stops a "Frankenstein" match where the digits are real but the
#      claimed meaning isn't.
#
# Every check below is scoped to ONE claim's span at a time (see
# _number_is_backed_by_a_single_claim / the per-claim loop in validate_asset)
# - never a blob formed by concatenating every claim_ref's span together.
# That matters once unit-context is involved: without per-claim scoping, a
# number from one claim's span and a "%" sign from a completely different
# claim's span could combine to falsely look like backing that no single
# piece of real evidence actually provides.
_NUMBER_CORE_PATTERN = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
# "%" gets NO trailing \b: it's a non-word character, so a literal \b placed
# right after it fails to match whenever "%" is followed by whitespace or
# end-of-string (both sides of that position would be \W) - which is
# precisely the common case ("40% of clients", "grew 40%."). Verified
# empirically before relying on it: a first version of this pattern DID
# require a trailing \b after "%" and silently never matched real percent
# signs at all, which would have made the unit-context check inert (falling
# back to a bare digit-boundary match with no unit awareness) rather than
# raising - the two regression tests below caught it. "percent"/"percentage"
# DO need the trailing \b, to reject a match inside "percentile".
_PERCENT_CONTEXT_PATTERN = re.compile(r"\s{0,2}(?:%|percent(?:age)?\b)", re.IGNORECASE)
_CONTEXT_LOOKAHEAD_CHARS = 12  # small window right after the number - "40" -> "%" or " percent"


class AssetValidationError(ValueError):
    """Raised when a draft fails validation, naming exactly what failed.
    Caught by the orchestrator (app/generation/__init__.py), which decides
    whether to retry or drop - never allowed to surface as an unhandled
    exception, and never converted into a validated=False GeneratedAsset
    that's returned anyway."""


def _number_occurrences(text: str) -> list[tuple[str, bool]]:
    """Every number in `text`, paired with whether IT is immediately
    followed by a percent marker in this same text - used so a percentage
    claim can only be backed by a span that also expresses it as a
    percentage, not just as a bare, differently-meant count."""
    occurrences = []
    for match in _NUMBER_CORE_PATTERN.finditer(text):
        remainder = text[match.end() : match.end() + _CONTEXT_LOOKAHEAD_CHARS]
        has_percent_context = bool(_PERCENT_CONTEXT_PATTERN.match(remainder))
        occurrences.append((match.group(), has_percent_context))
    return occurrences


def _number_is_backed_by_a_single_claim(number: str, wants_percent: bool, span_text: str) -> bool:
    """True if `number` appears in span_text with a real word boundary on
    both sides (never as a coincidental digit-substring of a longer number),
    and - when the generated text presented it as a percentage - the SAME
    occurrence in the span is also followed by a percent marker."""
    boundary_pattern = re.compile(rf"\b{re.escape(number)}\b")
    for match in boundary_pattern.finditer(span_text):
        if not wants_percent:
            return True
        remainder = span_text[match.end() : match.end() + _CONTEXT_LOOKAHEAD_CHARS]
        if _PERCENT_CONTEXT_PATTERN.match(remainder):
            return True
    return False


def validate_asset(draft: DraftAsset) -> GeneratedAsset:
    """
    Re-reads every referenced claim's CURRENT stored span text (never the
    text that was true when the claim was first extracted) and confirms,
    in order:

      1. (overview only) no claim_ref falls outside T1-T4 - an independent
         second guard on top of the generator's own filtering, in case a
         future generator change ever hands the model T5-T8 evidence by
         mistake. Proof from self-declared or peer-endorsed evidence must
         never reach a generated overview, regardless of which layer would
         otherwise have caught it.
      2. every claim_ref's span still reverifies against the source AS IT IS
         RIGHT NOW - if the underlying document changed or an index drifted
         since extraction, this draft is stale and must not be trusted.
      3. every number appearing in draft.text is backed by AT LEAST ONE
         claim_ref's CURRENT span text, checked with real word boundaries
         (never a bare digit-substring - "20" must not match inside "2019")
         and, when presented as a percentage, matching percent context in
         that SAME claim's span - see _number_is_backed_by_a_single_claim's
         docstring for the exact rule. Checked per claim, never against a
         blob of every claim_ref's span concatenated together.

    Deliberately biased toward over-rejection: a generic number like "24/7"
    or a year with no real backing will also get rejected here, and that's
    the correct trade-off - dropping good-but-unlucky content is safe,
    publishing one fabricated figure is not.

    Raises AssetValidationError naming what failed if any check fails.
    Returns a GeneratedAsset with validated=True ONLY when every check
    passes - this is the only function anywhere that sets that field.
    """
    if draft.kind == "overview":
        bad_tier_claims = [c for c in draft.claim_refs if c.evidence_tier not in PROOF_TIERS]
        if bad_tier_claims:
            raise AssetValidationError(
                f"overview draft references {len(bad_tier_claims)} claim(s) below T4 - "
                "proof must be drawn only from T1-T4 evidence"
            )

    current_spans: list[SourceSpan] = []
    for claim in draft.claim_refs:
        span = claim.source_span
        if span is None:
            raise AssetValidationError(f"referenced claim {claim.claim_text!r} has no source span")
        try:
            current_text = file_store.get_text(span.document_id)
        except FileNotFoundError as exc:
            raise AssetValidationError(
                f"stored document {span.document_id} for claim {claim.claim_text!r} is missing"
            ) from exc
        if not reverify_span(span, current_text):
            raise AssetValidationError(
                f"span for claim {claim.claim_text!r} no longer reverifies - source drifted"
            )
        current_spans.append(span)

    unbacked_numbers = sorted(
        {
            number
            for number, wants_percent in _number_occurrences(draft.text)
            if not any(
                _number_is_backed_by_a_single_claim(number, wants_percent, span.text)
                for span in current_spans
            )
        }
    )
    if unbacked_numbers:
        raise AssetValidationError(
            f"draft contains number(s) {unbacked_numbers} not backed (with a real word "
            "boundary and matching unit context) by any single referenced claim's "
            "current span text"
        )

    return GeneratedAsset(
        kind=draft.kind,
        text=draft.text,
        source_spans=current_spans,
        validated=True,
    )
