"""
The generator: drafts title and overview text against real evidence.

Built and tested TOGETHER with the validator (app/generation/validator.py) in
the same change, per CLAUDE.md - a generated draft cannot become a
GeneratedAsset without passing through validate_asset() first, and this
module is intentionally incapable of producing that final type at all (see
DraftAsset below).

Anthropic/Groq via the same `LLMClient` protocol the extractor already
defines (app/ingestion/extractor.py) - one method, complete(system, prompt),
so tests substitute a fake client exactly like ingestion's tests do. No live
API calls happen anywhere in this module's tests.

NOT IMPLEMENTED IN THIS SLICE - spec section 8's asset table lists four
assets (title, overview, case_study, proposal_draft); only the first two are
built here. `case_study` (problem, approach, result, evidence tier - "tier
shown to the user, not hidden") and `proposal_draft` (job description matched
to the strongest relevant evidence - "draft only, never auto-submitted") are
DEFERRED, not silently missing: no GeneratedAsset with kind="case_study" or
kind="proposal_draft" is ever produced anywhere in this codebase yet. Same
documented-gap pattern as app/scoring/dimensions.py's PROVISIONAL dimensions -
search this file for "NOT IMPLEMENTED" to find this note again. Extending
_DRAFT_BUILDERS in app/generation/__init__.py with two more (builder, kind)
pairs is the natural next step, following the exact same DraftAsset ->
validate_asset path already established for title/overview - no new
structural work needed, just two more generator functions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from app.ingestion.extractor import LLMClient
from app.schemas import Benchmark, Claim, EvidenceTier

# Spec section 8: the overview's proof section is drawn ONLY from T1-T4
# claims ("Why shipped work outranks certification" territory, but for
# generation rather than tier assignment). Reused by both the generator
# (which claims even get shown to the model) and the validator (an
# independent second check on the other side - see validator.py).
PROOF_TIERS = frozenset({EvidenceTier.T1, EvidenceTier.T2, EvidenceTier.T3, EvidenceTier.T4})

_NUMBER_PATTERN = re.compile(r"\d")


@dataclass
class DraftAsset:
    """
    An UNVALIDATED candidate asset - deliberately NOT a GeneratedAsset (see
    app/schemas/result.py). It has no `validated` field to fake, and no
    function in this module ever constructs a GeneratedAsset - only
    validate_asset() does, in validator.py. That is what makes "skipping
    validation" structurally impossible rather than a matter of remembering
    to call it: there is no code path from an LLM response to Result.generated
    that doesn't pass through validate_asset.

    `claim_refs` are the ONLY claims the validator is allowed to check this
    draft's numbers against - and critically, this list is built by OUR
    code (which evidence we chose to show the model), never by anything the
    model itself returns. That is how the overview keeps T5-T8 claims out of
    its proof section structurally: they are never in `claim_refs` because
    they were never in the evidence handed to the model in the first place,
    not because we trust the model to leave them out.
    """

    kind: str
    text: str
    claim_refs: list[Claim]


def _select_title_claims(claims: list[Claim], limit: int = 3) -> list[Claim]:
    """Best available OUTCOME evidence for the title: grounded claims whose
    proven span text contains a measurable (numeric) outcome, strongest
    evidence first. A title with no numeric backing at all has nothing
    fabricatable to protect against, but also nothing to feature - see
    generate_title_draft's None return when this is empty."""
    candidates = [
        c for c in claims if c.publishable and _NUMBER_PATTERN.search(c.source_span.text)
    ]
    return sorted(candidates, key=lambda c: c.effective_weight, reverse=True)[:limit]


def _select_overview_proof_claims(claims: list[Claim]) -> list[Claim]:
    """Spec section 8: "Proof section drawn only from T1 to T4 claims."
    Filtered here, before the model ever sees anything - not a post-hoc
    check on what the model wrote."""
    return [c for c in claims if c.publishable and c.evidence_tier in PROOF_TIERS]


class _TitleResponse(BaseModel):
    text: str | None = None


class _OverviewResponse(BaseModel):
    text: str | None = None


def _title_system_prompt(benchmark: Benchmark) -> str:
    return (
        f"You write a one-line professional title for a freelancer in the "
        f"niche '{benchmark.niche}'. Follow this exact structure: "
        f"{benchmark.title_formula}. Use ONLY the evidence given below - "
        f"never invent a role, outcome, or number that isn't in it. If the "
        f"evidence doesn't support a specific measurable outcome, omit the "
        f'outcome rather than inventing one. Return ONLY JSON matching '
        f'{{"text": str}}. No prose, no markdown fences - JSON only.'
    )


def _overview_system_prompt(benchmark: Benchmark) -> str:
    return (
        f"You write a freelancer profile overview for the niche "
        f"'{benchmark.niche}', {benchmark.overview_words_min}-"
        f"{benchmark.overview_words_max} words. Structure: a hook on the "
        f"buyer's problem, a proof section citing ONLY the evidence given "
        f"below, a short description of process, and a call to action. Use "
        f"ONLY the evidence given - never invent a number, outcome, or "
        f'skill that isn\'t in it. Return ONLY JSON matching {{"text": '
        f'str}}. No prose, no markdown fences - JSON only.'
    )


def generate_title_draft(
    client: LLMClient, claims: list[Claim], benchmark: Benchmark
) -> DraftAsset | None:
    """
    Returns None (nothing to draft, not an error) when there's no claim with
    a measurable outcome to build a title around - an empty result in the
    right shape, per convention, rather than a title with an invented number.
    """
    title_claims = _select_title_claims(claims)
    if not title_claims:
        return None

    evidence_block = "\n".join(f"- {c.source_span.text}" for c in title_claims)
    raw = client.complete(system=_title_system_prompt(benchmark), prompt=evidence_block)
    try:
        parsed = _TitleResponse.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        return None
    if not parsed.text:
        return None

    return DraftAsset(kind="title", text=parsed.text, claim_refs=title_claims)


def generate_overview_draft(
    client: LLMClient, claims: list[Claim], benchmark: Benchmark
) -> DraftAsset | None:
    """Returns None when there is no T1-T4 evidence to draw proof from at
    all - never generates an overview whose "proof" is entirely invented."""
    proof_claims = _select_overview_proof_claims(claims)
    if not proof_claims:
        return None

    evidence_block = "\n".join(f"- {c.source_span.text}" for c in proof_claims)
    raw = client.complete(system=_overview_system_prompt(benchmark), prompt=evidence_block)
    try:
        parsed = _OverviewResponse.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        return None
    if not parsed.text:
        return None

    return DraftAsset(kind="overview", text=parsed.text, claim_refs=proof_claims)
