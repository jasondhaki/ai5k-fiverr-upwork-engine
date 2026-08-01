"""
The pipeline orchestrator: the spine that runs input -> claims -> score ->
gaps -> generation -> result.

This is the walking skeleton: the whole path runs from a browser before
everything is real, so integration stops being a cliff at the end. Replace
one stub at a time with the real component; the contract between stages
never changes.

Build order for replacing stubs:
  1. extract_claims   (M1 - ingestion)                          REAL
  2. load_benchmark   (M2 - benchmark)                          REAL
  3. assign_tiers     (M2 - evidence rules)                     REAL
  4. score_profile    (M3 - seven dimensions + caps)            REAL
  5. rank_gaps        (M3 - gain/priority + blocking + dependencies)  REAL
  6. generate_assets  (M3 - generator + span validator, built together)  REAL
"""

from __future__ import annotations

import uuid
from typing import Callable

from app.evidence import assign_tiers as _assign_tiers
from app.evidence import load_benchmark as _load_benchmark
from app.generation import generate_assets as _generate_assets
from app.ingestion.extractor import status_reporting
from app.ingestion.router import route_input
from app.schemas import (
    Benchmark,
    BlockingItem,
    Claim,
    DimensionScore,
    Gap,
    GeneratedAsset,
    Result,
)
from app.scoring import find_skill_gaps as _find_skill_gaps
from app.scoring import rank_gaps as _rank_gaps
from app.scoring import score_profile as _score_profile
from app.storage.repository import Repository


class PipelineInput:
    """What goes in: a CV, a GitHub username, pasted Upwork profile text."""

    def __init__(
        self,
        niche: str,
        cv_bytes: bytes | None = None,
        github_username: str | None = None,
        upwork_text: str | None = None,
    ) -> None:
        self.niche = niche
        self.cv_bytes = cv_bytes
        self.github_username = github_username
        self.upwork_text = upwork_text


# --- REAL 1: ingestion -------------------------------------------------------
def extract_claims(
    inp: PipelineInput, on_status: Callable[..., None] | None = None
) -> list[Claim]:
    """Deterministic router dispatches each populated source (CV, GitHub,
    Upwork) to its parser and concatenates the resulting grounded claims (M1)."""
    return route_input(inp, on_status=on_status)


# --- REAL 2: benchmark --------------------------------------------------------
def load_benchmark(niche: str, version: str) -> Benchmark:
    """Load the versioned benchmark for this niche, by explicit (niche,
    version) - never "the latest". Fails loudly if that exact version
    doesn't exist (M2)."""
    return _load_benchmark(niche, version)


# --- REAL 3: evidence rules ---------------------------------------------------
def assign_tiers(claims: list[Claim]) -> list[Claim]:
    """Refine ingestion's provisional per-claim tiers using the fuller,
    rules-based table - never a model call (M2)."""
    return _assign_tiers(claims)


# --- REAL 4: scoring ---------------------------------------------------------
def score_profile(claims: list[Claim], benchmark: Benchmark) -> tuple[list[DimensionScore], float, bool]:
    """
    The seven dimension functions, then the evidence cap as an explicit
    final step (M3). Takes `benchmark` rather than the original stub's bare
    `niche: str` - the real dimension functions need required_terms,
    portfolio targets, and the rate band, not just the niche name. Return
    type is unchanged from the stub."""
    return _score_profile(claims, benchmark)


# --- REAL 5: gap ranking -------------------------------------------------------
def rank_gaps(
    claims: list[Claim], dimensions: list[DimensionScore], benchmark: Benchmark
) -> tuple[list[Gap], list[BlockingItem]]:
    """
    gain/priority + blocking + dependency gating + the balanced top five
    (M3). Takes `claims` (for blocking/dependency signals) and `benchmark`
    (for dimension_targets) - the original stub took neither, since it
    returned fixed fake data. Return type is unchanged from the stub."""
    return _rank_gaps(claims, dimensions, benchmark)


# --- REAL 6: generation --------------------------------------------------------
def generate_assets(claims: list[Claim], benchmark: Benchmark) -> tuple[list[GeneratedAsset], bool]:
    """
    Generator + span validator, built together - every returned asset has
    validated=True, guaranteed structurally (M3). Takes `benchmark` rather
    than the original stub's bare `niche: str` - the real generator needs
    the title formula and overview word counts, not just the niche name.

    Return type also changed from the stub: now (assets, incomplete)
    instead of a bare list. `incomplete` is True when a title/overview was
    attempted (real evidence existed) but never passed validation, so it's
    silently missing from `assets` - without this, that failure mode is
    indistinguishable from "nothing to generate yet" once it reaches
    Result.generated. See Result.generation_incomplete."""
    return _generate_assets(claims, benchmark)


# --- The spine --------------------------------------------------------------
def run_pipeline(
    inp: PipelineInput,
    benchmark_version: str = "2026-07",
    repository: Repository | None = None,
    on_status: Callable[..., None] | None = None,
    run_id: str | None = None,
) -> Result:
    """
    Runs the whole path. Every stage is now real.

    Mints `run_id` unconditionally, right at the top, before anything else
    runs - every Result this function returns is self-identifying, whether
    or not it ends up persisted. Persistence itself only happens when a
    `repository` is passed in (api.py passes the real singleton;
    the ~200+ existing tests that call run_pipeline directly - with no
    repository argument - keep working exactly as before, with zero DB
    access, since that's the default). When a repository IS given, the
    Claims and the Result are saved under that same run_id (via
    Repository.save_claims/save_result's `result_id` parameter), so a stored
    Result can always be traced back to the exact claim set that produced
    it - not just "some claims exist somewhere". See Repository.get_report.

    `run_id`, if given, is used instead of minting a fresh one - this is what
    lets the async /analyze endpoint (app/platform/api.py) know the run_id
    BEFORE the pipeline finishes: it mints one itself, registers it in the
    status store, redirects the browser to /analyze/{run_id} immediately, and
    only then starts this function as a background task with that same id -
    so the id in the URL and the id on the eventual Result are guaranteed to
    match. Every existing caller that doesn't pass one still gets a fresh id
    minted here, same as before.

    `on_status`, if given, is called with keyword-argument updates
    (stage=..., detail=..., progress_current=..., claims_found=..., etc.) as
    each stage starts or progresses - see app/platform/status.py's RunStatus
    for the full set of fields, and CLAUDE.md's "Swappable backends"-adjacent
    async-flow notes for why this is a plain callback rather than importing
    the status store here (pipeline.py has no business knowing HOW status is
    stored, only that something wants to hear about progress). Wrapping the
    whole run in status_reporting(on_status) makes that same callback
    available to GroqClient's rate-limit backoff too - see extractor.py.
    """
    run_id = run_id or uuid.uuid4().hex

    with status_reporting(on_status):
        claims = extract_claims(inp, on_status=on_status)

        if on_status is not None:
            on_status(stage="assigning_tiers", detail="Refining evidence tiers")
        claims = assign_tiers(claims)

        benchmark = load_benchmark(inp.niche, benchmark_version)

        if on_status is not None:
            on_status(stage="scoring", detail="Scoring profile and ranking gaps")
        dimensions, readiness, capped = score_profile(claims, benchmark)
        gaps, blocking = rank_gaps(claims, dimensions, benchmark)
        # A separate, UNSCORED report (spec section 5) - never a scoring
        # dimension, never consulted by score_profile/rank_gaps above, and
        # never a factor in `readiness`. Computed independently and attached
        # to the same Result purely for convenience; see Result.skill_gaps
        # and app/scoring/skill_gaps.py for why it can't quietly bleed into
        # scoring by construction (it isn't imported by either module).
        skill_gaps = _find_skill_gaps(claims, benchmark)

        if on_status is not None:
            on_status(stage="generating", detail="Generating title and overview")
        generated, generation_incomplete = generate_assets(claims, benchmark)

        provable = sum(1 for c in claims if c.publishable)

        result = Result(
            run_id=run_id,
            niche=inp.niche,
            # Stamp the version actually used - benchmark.version, not the raw
            # `benchmark_version` argument - so this can never drift from what
            # load_benchmark actually loaded and validated.
            benchmark_version=benchmark.version,
            readiness=readiness,
            capped=capped,
            cap_note="Capped at 30 until claims are proven" if capped else None,
            dimensions=dimensions,
            blocking=blocking,
            gaps=gaps,
            generated=generated,
            generation_incomplete=generation_incomplete,
            total_claims=len(claims),
            provable_claims=provable,
            skill_gaps=skill_gaps,
        )

        if repository is not None:
            repository.save_result(result, result_id=run_id)
            repository.save_claims(claims, result_id=run_id)

        if on_status is not None:
            on_status(
                stage="done",
                detail="Complete",
                claims_found=len(claims),
                claims_provable=provable,
            )

    return result
