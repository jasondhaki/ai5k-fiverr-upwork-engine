"""
Evidence: benchmark loading and rules-based tier assignment (M2).

    from app.evidence import load_benchmark, assign_tiers
"""

from app.evidence.benchmarks import BenchmarkNotFoundError, load_benchmark
from app.evidence.tiers import assign_tiers

__all__ = [
    "load_benchmark",
    "BenchmarkNotFoundError",
    "assign_tiers",
]
