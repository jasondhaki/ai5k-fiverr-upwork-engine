"""
The result record: everything that comes back from one run of the pipeline.

This is M3's output and M4's input - the handoff at the last seam. It carries
the capped readiness score, the seven dimensions broken out, blocking items
listed separately from ranked gaps, and generated text with source spans
attached so the validator's guarantee survives all the way to the page.

There is deliberately NO single headline percentage beyond the capped readiness
score. A single match number pushes people toward keyword stuffing; per-
dimension gaps with actions preserve the information that makes the report useful.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.claim import SourceSpan


class BlockingReason(str, Enum):
    """Why an item is a liability rather than an optimisation opportunity."""

    UNPROVEN_CLAIM = "unproven_claim"
    TOS_RISK = "tos_risk"
    MISSING_IDENTITY_VERIFICATION = "missing_identity_verification"


class DimensionScore(BaseModel):
    """One of the seven scoring dimensions."""

    name: str
    score: float = Field(..., ge=0.0, le=100.0)
    weight: float = Field(..., ge=0.0, le=1.0, description="Dimension weight from config")
    detail: Optional[str] = Field(
        None, description="Human-readable 'you now' summary, e.g. '2 of 11 claims proven'"
    )


class BlockingItem(BaseModel):
    """
    A liability that leaves the ranking and goes to a fix-before-publishing
    list, regardless of what its return on effort would have been.
    """

    reason: BlockingReason
    description: str
    dimension: Optional[str] = None


class Gap(BaseModel):
    """
    One ranked improvement opportunity. priority is points-per-hour:

        gain     = weight * (target - current) * efficacy
        priority = gain / max(effort_hours, 0.5)

    target is the benchmark value, not a perfect 100. efficacy comes from a
    versioned lookup table, never re-estimated per run.
    """

    dimension: str
    current: float = Field(..., ge=0.0, le=100.0)
    target: float = Field(..., ge=0.0, le=100.0)
    efficacy: float = Field(..., ge=0.0, le=1.0)
    effort_hours: float = Field(..., gt=0.0)
    gain: float = Field(..., ge=0.0)
    priority: float = Field(..., ge=0.0)


class GeneratedAsset(BaseModel):
    """
    A generated title or overview. Every number in `text` must be traceable to
    one of `source_spans`. The validator enforces this before the asset is
    allowed into a Result; an asset that fails validation never gets here.
    """

    kind: str = Field(..., description="'title' | 'overview' | 'case_study' | 'proposal_draft'")
    text: str
    source_spans: list[SourceSpan] = Field(
        default_factory=list,
        description="Every span backing a number or claim in the text",
    )
    validated: bool = Field(
        False, description="True only after the span validator has passed it"
    )


class Result(BaseModel):
    """The complete scored output for one profile run."""

    run_id: Optional[str] = Field(
        None,
        description="Stable identifier minted at the start of run_pipeline, shared with "
        "the Claims persisted alongside this Result (passed as `result_id` to "
        "Repository.save_result/save_claims - see app/storage/repository.py). None when "
        "a Result was constructed outside run_pipeline (e.g. directly in a test). This is "
        "what lets a stored Result always be traced back to the exact claim set that "
        "produced it, not just 'some claims exist somewhere' - see Repository.get_report.",
    )
    niche: str
    benchmark_version: str = Field(
        ..., description="Which benchmark version produced this score"
    )

    readiness: float = Field(..., ge=0.0, le=100.0, description="Overall score after caps")
    capped: bool = Field(
        ..., description="True if a cap was applied (thin evidence). Drives the UI framing."
    )
    cap_note: Optional[str] = Field(
        None, description="e.g. 'Capped at 30 until claims are proven'"
    )

    dimensions: list[DimensionScore]
    blocking: list[BlockingItem] = Field(default_factory=list)
    gaps: list[Gap] = Field(
        default_factory=list, description="The balanced top five, already assembled"
    )
    generated: list[GeneratedAsset] = Field(default_factory=list)
    generation_incomplete: bool = Field(
        False,
        description="True if at least one expected generated asset (title/overview) was "
        "attempted - real evidence existed to draft from - but never passed validation "
        "within the retry budget, so it's missing from `generated`. Distinct from simply "
        "having no evidence to draft from at all, which leaves this False. A diagnosability "
        "signal only, not user-facing messaging: without it, a user seeing no generated "
        "title has no way to tell 'nothing to generate yet' from 'generation kept failing'.",
    )

    # Provenance counts for the UI's opening framing ("you have 11 claims and
    # can prove 2"), which leads with position and route, never a bare figure.
    total_claims: int = 0
    provable_claims: int = 0
