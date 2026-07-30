"""
Generation: generator + validator, built and tested together (M3, spec
section 8).

    from app.generation import generate_assets

Policy for a draft that fails validation (deliberately chosen, documented
here since it's a judgment call the brief left open): retry generation
EXACTLY ONCE, mirroring the extractor's own RETRY_ATTEMPTS=2 policy (one
initial try, one retry) for consistency across the codebase. If the retry
also fails validation, the asset is DROPPED from the output entirely - never
returned with validated=False and the fabricated text still attached. Spec
section 8 is explicit that a number which can't be traced "does not reach
the draft" at all; there is no partial-credit or degraded-but-shown state
for generated content the way there is for claims (a coaching prompt has no
generation-layer equivalent - there's nothing to coach the user about a
draft their own evidence didn't support).
"""

from __future__ import annotations

import logging

from app.generation.generator import (
    generate_overview_draft,
    generate_title_draft,
)
from app.generation.validator import AssetValidationError, validate_asset
from app.ingestion.extractor import LLMClient, build_default_client
from app.schemas import Benchmark, Claim, GeneratedAsset

logger = logging.getLogger(__name__)

RETRY_ATTEMPTS = 2  # one initial try + one retry on a validation failure

_DRAFT_BUILDERS = {
    "title": generate_title_draft,
    "overview": generate_overview_draft,
}


def _validated_or_none(
    kind: str, builder, client: LLMClient, claims: list[Claim], benchmark: Benchmark
) -> tuple[GeneratedAsset | None, bool]:
    """
    Returns (asset_or_none, failed_validation).

    `failed_validation` is True ONLY when a draft was actually produced -
    real evidence existed to build one from - but never passed validation
    within RETRY_ATTEMPTS. It is deliberately False when the builder simply
    found no suitable evidence to draft from at all (draft is None every
    attempt): that's an expected, unremarkable empty result (spec's "empty
    output in the right shape is fine"), not a failure worth signaling -
    see generate_assets' `incomplete` return value, which this distinction
    feeds directly.
    """
    attempted = False
    for attempt in range(RETRY_ATTEMPTS):
        draft = builder(client, claims, benchmark)
        if draft is None:
            continue  # generator found no suitable evidence to draft from at all
        attempted = True
        try:
            return validate_asset(draft), False
        except AssetValidationError as exc:
            logger.warning(
                "%s draft failed validation (attempt %d/%d): %s",
                kind, attempt + 1, RETRY_ATTEMPTS, exc,
            )
            continue
    if attempted:
        logger.warning("%s: no validated draft after %d attempt(s) - dropped", kind, RETRY_ATTEMPTS)
    return None, attempted


def generate_assets(
    claims: list[Claim], benchmark: Benchmark, client: LLMClient | None = None
) -> tuple[list[GeneratedAsset], bool]:
    """
    Generate title + overview against real evidence. Every returned
    GeneratedAsset has validated=True - guaranteed structurally, since
    validate_asset() is the only function anywhere that constructs one (see
    app/generation/validator.py). An asset that never validates within
    RETRY_ATTEMPTS is dropped, not included unvalidated.

    Returns (assets, incomplete). `incomplete` is True if AT LEAST ONE
    expected asset kind was attempted (real evidence existed) but never
    validated - the signal a real report could otherwise go silently
    missing a title/overview with no way to tell "nothing to generate yet"
    from "generation kept failing" (see Result.generation_incomplete, and
    the per-kind log line at the point of failure above).
    """
    llm_client = client or build_default_client()
    assets: list[GeneratedAsset] = []
    incomplete = False
    for kind, builder in _DRAFT_BUILDERS.items():
        asset, failed = _validated_or_none(kind, builder, llm_client, claims, benchmark)
        if asset is not None:
            assets.append(asset)
        if failed:
            incomplete = True
    return assets, incomplete


__all__ = [
    "generate_assets",
    "AssetValidationError",
    "validate_asset",
]
