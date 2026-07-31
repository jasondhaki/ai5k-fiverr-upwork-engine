"""
Repository round-trip tests. Fast-suite tests run against an in-memory SQLite
engine created fresh per test - never the app's real DATABASE_URL-backed
singleton - so this file needs no network access and no Neon credentials.

`db`-marked tests at the bottom run the same round trips against the real
Neon database (DATABASE_URL from .env) to prove the JSONB/pgvector-specific
paths actually work on Postgres, not just SQLite's JSON fallback. Skipped by
default (see pyproject.toml's `db` marker); run deliberately with
`pytest -m db`.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.schemas import (
    Benchmark,
    BlockingItem,
    BlockingReason,
    Claim,
    DimensionScore,
    EvidenceTier,
    Gap,
    GeneratedAsset,
    RateBand,
    Result,
    SourceSpan,
    SourceType,
)
from app.storage.models import Base
from app.storage.repository import SqlAlchemyRepository


@pytest.fixture
def repo() -> SqlAlchemyRepository:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return SqlAlchemyRepository(session_factory=session_factory)


def _sample_claim() -> Claim:
    span = SourceSpan(document_id="doc-1", start_index=0, end_index=10, text="grew sales")
    return Claim(
        claim_text="grew sales 20%",
        skill_ids=["sales_ops"],
        source_type=SourceType.UPWORK_TEXT,
        source_span=span,
        evidence_tier=EvidenceTier.T1,
        weight=1.0,
        observed_date=date(2025, 3, 1),
        recency_factor=0.9,
        tier_rule="client_verified_outcome",
    )


def _sample_result() -> Result:
    return Result(
        niche="SMB workflow automation",
        benchmark_version="2026-07",
        readiness=57.0,
        capped=False,
        dimensions=[
            DimensionScore(name="evidence_quality", score=40.0, weight=0.2, detail="1 of 1 proven")
        ],
        blocking=[
            BlockingItem(
                reason=BlockingReason.UNPROVEN_CLAIM,
                description="one claim has no source span",
                dimension="evidence_quality",
            )
        ],
        gaps=[
            Gap(
                dimension="evidence_quality",
                current=40.0,
                target=80.0,
                efficacy=0.5,
                effort_hours=2.0,
                gain=4.0,
                priority=2.0,
            )
        ],
        generated=[GeneratedAsset(kind="title", text="Sales Ops Specialist", validated=True)],
        generation_incomplete=True,
        total_claims=3,
        provable_claims=1,
    )


def _sample_benchmark() -> Benchmark:
    return Benchmark(
        niche="SMB workflow automation",
        version="2026-07-test",
        required_terms=["automation", "n8n"],
        benchmark_topics=["workflow automation", "SMB tooling"],
        title_formula="[role] - [vertical] - [outcome]",
        overview_words_min=80,
        overview_words_max=150,
        portfolio_min_items=3,
        portfolio_min_quantified=1,
        rate_band=RateBand(low=40.0, high=90.0, justifying_tiers=["T1", "T2"]),
        dimension_targets={"evidence_quality": 80.0},
    )


def test_save_and_get_result_round_trips_every_field(repo: SqlAlchemyRepository):
    result = _sample_result()
    result_id = repo.save_result(result)

    fetched = repo.get_result(result_id)

    assert fetched is not None
    assert fetched.niche == result.niche
    assert fetched.readiness == result.readiness
    assert fetched.generation_incomplete is True
    assert fetched.dimensions[0].name == "evidence_quality"
    assert fetched.blocking[0].reason == BlockingReason.UNPROVEN_CLAIM
    assert fetched.gaps[0].priority == 2.0
    assert fetched.generated[0].kind == "title"


def test_get_result_returns_none_for_an_unknown_id(repo: SqlAlchemyRepository):
    assert repo.get_result("does-not-exist") is None


def test_save_result_honors_an_explicit_result_id(repo: SqlAlchemyRepository):
    """run_pipeline mints a run_id up front and passes it in explicitly so
    the Claims saved alongside share it - save_result must use that id
    verbatim, not silently mint its own."""
    returned_id = repo.save_result(_sample_result(), result_id="explicit-run-id")

    assert returned_id == "explicit-run-id"
    assert repo.get_result("explicit-run-id") is not None


def test_get_result_round_trips_run_id_onto_the_result(repo: SqlAlchemyRepository):
    result_id = repo.save_result(_sample_result(), result_id="run-abc")
    fetched = repo.get_result(result_id)
    assert fetched.run_id == "run-abc"


def test_save_and_get_claims_round_trips_the_source_span_and_date(
    repo: SqlAlchemyRepository,
):
    result_id = repo.save_result(_sample_result())
    repo.save_claims([_sample_claim()], result_id=result_id)

    fetched = repo.get_claims(result_id)

    assert len(fetched) == 1
    claim = fetched[0]
    assert claim.claim_text == "grew sales 20%"
    assert claim.source_span is not None
    assert claim.source_span.text == "grew sales"
    assert claim.evidence_tier == EvidenceTier.T1
    assert claim.observed_date == date(2025, 3, 1)
    assert claim.publishable is True


def test_get_claims_only_returns_claims_linked_to_that_result(repo: SqlAlchemyRepository):
    result_a = repo.save_result(_sample_result())
    result_b = repo.save_result(_sample_result())
    repo.save_claims([_sample_claim()], result_id=result_a)

    assert len(repo.get_claims(result_a)) == 1
    assert repo.get_claims(result_b) == []


def test_get_report_returns_the_result_and_its_claims_together(repo: SqlAlchemyRepository):
    result_id = repo.save_result(_sample_result(), result_id="run-with-claims")
    repo.save_claims([_sample_claim()], result_id=result_id)

    report = repo.get_report(result_id)

    assert report is not None
    result, claims = report
    assert result.run_id == result_id
    assert len(claims) == 1
    assert claims[0].claim_text == "grew sales 20%"


def test_get_report_returns_none_when_no_result_was_ever_saved(repo: SqlAlchemyRepository):
    assert repo.get_report("never-saved") is None


def test_get_report_returns_an_empty_claims_list_when_none_were_linked(
    repo: SqlAlchemyRepository,
):
    """A Result can exist with no linked Claims (e.g. save_claims was never
    called) - that's an empty list, not a missing report."""
    result_id = repo.save_result(_sample_result())

    report = repo.get_report(result_id)

    assert report is not None
    _, claims = report
    assert claims == []


def test_save_and_get_benchmark_round_trips(repo: SqlAlchemyRepository):
    benchmark = _sample_benchmark()
    repo.save_benchmark(benchmark)

    fetched = repo.get_benchmark(benchmark.niche, benchmark.version)

    assert fetched is not None
    assert fetched.rate_band.low == 40.0
    assert fetched.dimension_targets == {"evidence_quality": 80.0}
    assert fetched.required_terms == ["automation", "n8n"]


def test_save_benchmark_upserts_rather_than_duplicating(repo: SqlAlchemyRepository):
    """Benchmarks are immutable per (niche, version) - re-saving the same
    pair replaces the row instead of erroring or accumulating a duplicate."""
    benchmark = _sample_benchmark()
    repo.save_benchmark(benchmark)
    updated = benchmark.model_copy(update={"portfolio_min_items": 5})
    repo.save_benchmark(updated)

    fetched = repo.get_benchmark(benchmark.niche, benchmark.version)
    assert fetched.portfolio_min_items == 5


def test_get_benchmark_returns_none_for_an_unknown_niche_version(repo: SqlAlchemyRepository):
    assert repo.get_benchmark("no such niche", "0000-00") is None


# --- db-marked: real Neon Postgres, run deliberately -------------------------


@pytest.mark.db
def test_result_round_trip_against_real_postgres():
    """Proves the JSONB columns (not SQLite's JSON fallback) and the live
    Alembic-migrated schema actually work end to end on Neon."""
    from app.storage.db import SessionLocal, engine
    from app.storage.models import ResultRecord

    repo = SqlAlchemyRepository(session_factory=SessionLocal)
    result_id = repo.save_result(_sample_result())
    try:
        fetched = repo.get_result(result_id)
        assert fetched is not None
        assert fetched.generated[0].kind == "title"
    finally:
        with engine.connect() as conn:
            conn.execute(ResultRecord.__table__.delete().where(ResultRecord.id == result_id))
            conn.commit()
