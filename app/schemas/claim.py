"""
The claim record: the atomic unit of the entire system.

THIS FILE IS FROZEN. After the schema-lock checkpoint, you may ADD optional
fields, but you may NEVER rename or remove one. Additions don't break other
people's code; renames and removals do. This rule is what keeps four parallel
workstreams (or four Claude Code sessions) from colliding.

The governing invariant of the whole product lives here: a claim with no
source_span is never publishable. If the evidence cannot be pointed at, the
claim does not go on the profile - it becomes a coaching prompt instead.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class EvidenceTier(str, Enum):
    """
    Eight tiers, ordered strongest to weakest. A skill takes its STRONGEST
    evidence, never the sum of weak evidence. Weights live in config/weights,
    not here, so they can be tuned without touching the schema.
    """

    T1 = "T1"  # Client-verified outcome, named in a review or testimonial
    T2 = "T2"  # Project demonstrated: deployed repo, live app, published model
    T3 = "T3"  # Platform-assessed: our own structured skills assessment
    T4 = "T4"  # Certification, proctored and verifiable with the issuer
    T5 = "T5"  # Certification, badge only or self-paced course
    T6 = "T6"  # Employer-confirmed: work history where the skill was used
    T7 = "T7"  # Peer-endorsed: recommendation from a colleague
    T8 = "T8"  # Self-declared, with no corroboration


class SourceType(str, Enum):
    """Where a claim was extracted from. Drives tier assignment (rules-based)."""

    CV = "cv"
    GITHUB_REPO = "github_repo"
    UPWORK_TEXT = "upwork_text"
    LINKEDIN_EXPORT = "linkedin_export"
    PORTFOLIO_SITE = "portfolio_site"
    HUGGINGFACE = "huggingface"
    DEMO_VIDEO = "demo_video"
    ARTICLE = "article"
    ONBOARDING_FORM = "onboarding_form"


class SourceSpan(BaseModel):
    """
    A pointer back to the exact place a claim came from.

    The model returns character indices into the source text; post-processing
    re-slices the literal substring from those indices. `text` is that
    re-extracted substring - it is NEVER paraphrased by a model. This is what
    makes grounding real rather than aspirational.

    `document_id` lets the validator re-read the original file at generation
    time, which is why original files are retained in object storage.
    """

    document_id: str = Field(..., description="ID of the stored source file")
    start_index: int = Field(..., ge=0, description="Char offset, inclusive")
    end_index: int = Field(..., gt=0, description="Char offset, exclusive")
    text: str = Field(..., description="Literal substring re-sliced from source")
    locator: Optional[str] = Field(
        None, description="Human-readable pointer, e.g. 'page 2 line 14' or 'repo rag-pipeline'"
    )

    @model_validator(mode="after")
    def _indices_consistent(self) -> "SourceSpan":
        if self.end_index <= self.start_index:
            raise ValueError("end_index must be greater than start_index")
        if len(self.text) != self.end_index - self.start_index:
            raise ValueError(
                "span text length does not match index range - "
                "the substring was not re-sliced from the source correctly"
            )
        return self


class Claim(BaseModel):
    """
    One piece of evidence about one freelancer.

    An empty list of claims is an acceptable output from a parser. A claim in
    the wrong shape is not. Every field except source_span is populated at
    extraction time; publishable is derived, never hand-set.
    """

    claim_text: str = Field(..., description="What is claimed, in plain language")
    skill_ids: list[str] = Field(default_factory=list, description="Normalized skill tags")
    source_type: SourceType
    source_span: Optional[SourceSpan] = Field(
        None,
        description="Origin pointer. If None, the claim is NOT publishable and "
        "becomes a coaching prompt.",
    )
    evidence_tier: EvidenceTier
    weight: float = Field(..., ge=0.0, le=1.0, description="Tier weight, from config")
    observed_date: Optional[date] = Field(
        None, description="When the evidence was created, e.g. last commit date"
    )
    recency_factor: float = Field(
        1.0, ge=0.0, le=1.0, description="Discount for age; 1.0 = current"
    )
    tier_rule: Optional[str] = Field(
        None,
        description="Which named rule in assign_tiers's table decided evidence_tier "
        "(e.g. 'client_verified_outcome'). None before that stage runs. Every tier "
        "must be traceable to a rule name, never just 'plausible in the aggregate'.",
    )

    @property
    def publishable(self) -> bool:
        """
        THE RULE THE WHOLE SYSTEM RESTS ON.

        A claim is publishable if and only if it can be traced to a source
        span. This is a derived property, not a stored field, so it can never
        drift out of sync with the evidence. Nothing downstream may publish a
        claim for which this returns False.
        """
        return self.source_span is not None

    @property
    def effective_weight(self) -> float:
        """Tier weight discounted by recency. Used by the evidence dimension."""
        return self.weight * self.recency_factor
