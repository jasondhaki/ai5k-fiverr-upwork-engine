"""
The benchmark record: what a winning profile looks like in one niche.

Benchmarks are VERSIONED and IMMUTABLE. A score computed in March must always
be explainable by the March benchmark, so the scorer reads a specific version -
it never reads "the latest". The version string is part of the identity of the
record.

For the sprint slice this is one hand-written file for one niche. For the full
system the anchor track populates it automatically. Both produce this exact
shape, which is the whole point: the hand-built and automated benchmarks are
interchangeable to everything downstream.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RateBand(BaseModel):
    """Where rates cluster for a given evidence level, in USD/hour."""

    low: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    justifying_tiers: list[str] = Field(
        default_factory=lambda: ["T1", "T2"],
        description="Evidence tiers that justify this band",
    )


class Benchmark(BaseModel):
    """
    Aggregate patterns extracted from top profiles in one niche. The underlying
    profiles are DISCARDED after extraction - only these statistics are kept.
    That is both the safer position on personal data and the more useful one,
    since the scorer only ever consumes the aggregates.
    """

    niche: str = Field(..., description="e.g. 'SMB workflow automation'")
    version: str = Field(..., description="e.g. '2026-07'. Immutable once written.")

    required_terms: list[str] = Field(
        ..., description="Terms the benchmark marks as required for this niche"
    )
    benchmark_topics: list[str] = Field(
        default_factory=list,
        description="Topics for semantic coverage matching (the 30% half of keyword score)",
    )

    title_formula: str = Field(..., description="e.g. '[role] - [vertical] - [outcome]'")
    overview_words_min: int = Field(..., gt=0)
    overview_words_max: int = Field(..., gt=0)
    portfolio_min_items: int = Field(..., ge=0)
    portfolio_min_quantified: int = Field(
        0, ge=0, description="Minimum portfolio items that must carry numbers"
    )

    rate_band: RateBand

    # Per-dimension targets: the value the top tier actually reaches in this
    # niche, NOT a perfect 100. Keyed by dimension name. The gap ranker aims
    # profiles at these targets, not at perfection.
    dimension_targets: dict[str, float] = Field(
        ..., description="dimension name -> target score (0-100)"
    )

    model_config = {"frozen": True}  # immutable once constructed
