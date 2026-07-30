"""
Benchmark loading tests. The real behavior that matters: a score must always
be explainable by a NAMED version - there is no "load the latest for this
niche" path, and a request that doesn't match exactly what's on disk fails
loudly rather than silently substituting something else.
"""

from __future__ import annotations

import json

import pytest

from app.evidence.benchmarks import BenchmarkNotFoundError, load_benchmark
from app.schemas import Benchmark

REAL_NICHE = "SMB workflow automation"
REAL_VERSION = "2026-07"


def test_load_benchmark_returns_the_real_versioned_file():
    benchmark = load_benchmark(REAL_NICHE, REAL_VERSION)
    assert isinstance(benchmark, Benchmark)
    assert benchmark.niche == REAL_NICHE
    assert benchmark.version == REAL_VERSION
    assert "n8n" in benchmark.required_terms


def test_load_benchmark_fails_loudly_for_a_missing_version():
    with pytest.raises(BenchmarkNotFoundError):
        load_benchmark(REAL_NICHE, "1999-01")


def test_load_benchmark_fails_loudly_for_an_unknown_niche():
    with pytest.raises(BenchmarkNotFoundError):
        load_benchmark("Underwater basket weaving", REAL_VERSION)


def _write_benchmark(directory, niche: str, version: str, **overrides) -> None:
    slug = niche.lower().replace(" ", "_")
    data = {
        "niche": niche,
        "version": version,
        "required_terms": ["x"],
        "title_formula": "[role] - [vertical] - [outcome]",
        "overview_words_min": 100,
        "overview_words_max": 200,
        "portfolio_min_items": 1,
        "rate_band": {"low": 10.0, "high": 20.0},
        "dimension_targets": {"positioning": 80.0},
        **overrides,
    }
    (directory / f"{slug}_{version}.json").write_text(json.dumps(data), encoding="utf-8")


def test_load_benchmark_never_falls_back_to_the_latest_version(tmp_path):
    _write_benchmark(tmp_path, "Test Niche", "2026-01")
    _write_benchmark(tmp_path, "Test Niche", "2026-06")

    # explicitly requesting the OLDER version must return exactly that one,
    # never the newer file that also exists right next to it
    benchmark = load_benchmark("Test Niche", "2026-01", benchmarks_dir=tmp_path)
    assert benchmark.version == "2026-01"


def test_load_benchmark_rejects_a_file_whose_internal_niche_does_not_match(tmp_path):
    # filename says "requested_niche_2026-01.json" but the file's own `niche`
    # field disagrees - a stale copy-paste on disk must not silently score
    # under the wrong label
    slug_path = tmp_path / "requested_niche_2026-01.json"
    data = {
        "niche": "A completely different niche",
        "version": "2026-01",
        "required_terms": ["x"],
        "title_formula": "[role] - [vertical] - [outcome]",
        "overview_words_min": 100,
        "overview_words_max": 200,
        "portfolio_min_items": 1,
        "rate_band": {"low": 10.0, "high": 20.0},
        "dimension_targets": {"positioning": 80.0},
    }
    slug_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(BenchmarkNotFoundError):
        load_benchmark("Requested Niche", "2026-01", benchmarks_dir=tmp_path)


def test_load_benchmark_rejects_a_file_whose_internal_version_does_not_match(tmp_path):
    _write_benchmark(tmp_path, "Test Niche", "2026-01")
    # rename the file to claim a version its contents don't actually have
    (tmp_path / "test_niche_2026-01.json").rename(tmp_path / "test_niche_2026-02.json")

    with pytest.raises(BenchmarkNotFoundError):
        load_benchmark("Test Niche", "2026-02", benchmarks_dir=tmp_path)
