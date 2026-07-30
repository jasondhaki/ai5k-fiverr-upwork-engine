"""
The three frozen record shapes. Everything imports from here.

    from app.schemas import Claim, Benchmark, Result

These are the two-hour blocking task made permanent. Add optional fields if you
must; never rename or remove one.
"""

from app.schemas.benchmark import Benchmark, RateBand
from app.schemas.claim import Claim, EvidenceTier, SourceSpan, SourceType
from app.schemas.result import (
    BlockingItem,
    BlockingReason,
    DimensionScore,
    Gap,
    GeneratedAsset,
    Result,
)

__all__ = [
    "Claim",
    "SourceSpan",
    "SourceType",
    "EvidenceTier",
    "Benchmark",
    "RateBand",
    "Result",
    "DimensionScore",
    "BlockingItem",
    "BlockingReason",
    "Gap",
    "GeneratedAsset",
]
