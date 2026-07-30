"""
Benchmark loading: read a specific (niche, version) from data/benchmarks/.

Benchmarks are versioned and immutable (see app/schemas/benchmark.py) - a
score computed against the 2026-07 benchmark must always be explainable by
the 2026-07 file, even after a 2026-08 benchmark exists. There is deliberately
no "load the latest for this niche" path: every caller names an exact
version, and a missing one fails loudly rather than silently substituting
something else.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.schemas import Benchmark

DEFAULT_BENCHMARKS_DIR = Path("data/benchmarks")


class BenchmarkNotFoundError(FileNotFoundError):
    """Raised when the exact (niche, version) requested has no matching file -
    or the file that does exist doesn't actually contain what was asked for.
    Never swallowed: a typo'd version should fail loudly, not silently score
    against whatever happened to be on disk."""


def _slugify(niche: str) -> str:
    """'SMB workflow automation' -> 'smb_workflow_automation' - matches the
    on-disk filename convention. Model-free: a plain, deterministic transform,
    not a lookup that could drift from the actual filenames on disk."""
    return re.sub(r"[^a-z0-9]+", "_", niche.lower()).strip("_")


def load_benchmark(
    niche: str, version: str, benchmarks_dir: str | Path = DEFAULT_BENCHMARKS_DIR
) -> Benchmark:
    """
    Load the benchmark for exactly (niche, version). Fails loudly via
    BenchmarkNotFoundError if:
      - no file matches the expected {slug(niche)}_{version}.json name, or
      - the file exists but its own niche/version fields don't match what was
        requested (a stale rename or copy-paste on disk should never silently
        score under the wrong label).
    """
    directory = Path(benchmarks_dir)
    filename = f"{_slugify(niche)}_{version}.json"
    path = directory / filename

    if not path.is_file():
        raise BenchmarkNotFoundError(
            f"No benchmark found for niche={niche!r} version={version!r} "
            f"(expected {path}). There is no 'latest' fallback - request an "
            f"exact version that exists."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    benchmark = Benchmark.model_validate(data)

    if benchmark.niche != niche or benchmark.version != version:
        raise BenchmarkNotFoundError(
            f"{path} contains niche={benchmark.niche!r} version={benchmark.version!r}, "
            f"but niche={niche!r} version={version!r} was requested - refusing to "
            f"score under a mismatched label."
        )

    return benchmark
