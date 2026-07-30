"""
Persistence behind a Repository interface - the same way app/storage/store.py
already abstracts file storage behind the FileStore protocol. Nothing outside
this module (and app/storage/models.py, which it alone imports) touches
SQLAlchemy directly: api.py and pipeline.py depend only on `Repository` and
the `repository` singleton below.

No table is created by this module. `SqlAlchemyRepository` only ever opens a
connection when a method actually runs a query - constructing the singleton
at import time (even with a real Neon DATABASE_URL in .env) makes zero network
calls, which is what keeps the fast test suite import-safe. Tables are
created by Alembic migrations (see alembic/) against a real database, or by
an explicit `Base.metadata.create_all(engine)` in test fixtures against an
in-memory SQLite engine (tests/test_repository.py).
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Protocol

from sqlalchemy.orm import Session, sessionmaker

from app.schemas import (
    Benchmark,
    BlockingItem,
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
from app.storage.db import SessionLocal
from app.storage.models import BenchmarkRecord, ClaimRecord, ResultRecord


class Repository(Protocol):
    """The only persistence contract the rest of the system knows about."""

    def save_result(self, result: Result) -> str:
        """Persist one pipeline run, return its new result_id."""
        ...

    def get_result(self, result_id: str) -> Result | None:
        ...

    def save_claims(self, claims: list[Claim], *, result_id: str | None = None) -> list[str]:
        """Persist claims, optionally linked to a result_id. Returns new claim ids."""
        ...

    def get_claims(self, result_id: str) -> list[Claim]:
        ...

    def save_benchmark(self, benchmark: Benchmark) -> None:
        """Upsert by (niche, version) - benchmarks are immutable per version,
        so a re-save of the same (niche, version) replaces the row rather
        than accumulating duplicates."""
        ...

    def get_benchmark(self, niche: str, version: str) -> Benchmark | None:
        ...


def _dump_span(span: SourceSpan | None) -> dict | None:
    return span.model_dump(mode="json") if span is not None else None


class SqlAlchemyRepository:
    """Real implementation. Each method opens and closes its own short-lived
    Session - callers never see a SQLAlchemy object, only Pydantic models."""

    def __init__(self, session_factory: sessionmaker[Session] = SessionLocal) -> None:
        self._session_factory = session_factory

    # --- results ---------------------------------------------------------
    def save_result(self, result: Result) -> str:
        result_id = uuid.uuid4().hex
        with self._session_factory() as session:
            session.add(
                ResultRecord(
                    id=result_id,
                    niche=result.niche,
                    benchmark_version=result.benchmark_version,
                    readiness=result.readiness,
                    capped=result.capped,
                    cap_note=result.cap_note,
                    dimensions=[d.model_dump(mode="json") for d in result.dimensions],
                    blocking=[b.model_dump(mode="json") for b in result.blocking],
                    gaps=[g.model_dump(mode="json") for g in result.gaps],
                    generated=[g.model_dump(mode="json") for g in result.generated],
                    generation_incomplete=result.generation_incomplete,
                    total_claims=result.total_claims,
                    provable_claims=result.provable_claims,
                )
            )
            session.commit()
        return result_id

    def get_result(self, result_id: str) -> Result | None:
        with self._session_factory() as session:
            row = session.get(ResultRecord, result_id)
            if row is None:
                return None
            return Result(
                niche=row.niche,
                benchmark_version=row.benchmark_version,
                readiness=row.readiness,
                capped=row.capped,
                cap_note=row.cap_note,
                dimensions=[DimensionScore.model_validate(d) for d in row.dimensions],
                blocking=[BlockingItem.model_validate(b) for b in row.blocking],
                gaps=[Gap.model_validate(g) for g in row.gaps],
                generated=[GeneratedAsset.model_validate(g) for g in row.generated],
                generation_incomplete=row.generation_incomplete,
                total_claims=row.total_claims,
                provable_claims=row.provable_claims,
            )

    # --- claims ------------------------------------------------------------
    def save_claims(self, claims: list[Claim], *, result_id: str | None = None) -> list[str]:
        ids: list[str] = []
        with self._session_factory() as session:
            for claim in claims:
                claim_id = uuid.uuid4().hex
                session.add(
                    ClaimRecord(
                        id=claim_id,
                        result_id=result_id,
                        claim_text=claim.claim_text,
                        skill_ids=list(claim.skill_ids),
                        source_type=claim.source_type.value,
                        source_span=_dump_span(claim.source_span),
                        evidence_tier=claim.evidence_tier.value,
                        weight=claim.weight,
                        observed_date=(
                            claim.observed_date.isoformat() if claim.observed_date else None
                        ),
                        recency_factor=claim.recency_factor,
                        tier_rule=claim.tier_rule,
                    )
                )
                ids.append(claim_id)
            session.commit()
        return ids

    def get_claims(self, result_id: str) -> list[Claim]:
        with self._session_factory() as session:
            rows = session.query(ClaimRecord).filter_by(result_id=result_id).all()
            return [
                Claim(
                    claim_text=row.claim_text,
                    skill_ids=row.skill_ids,
                    source_type=SourceType(row.source_type),
                    source_span=(
                        SourceSpan.model_validate(row.source_span) if row.source_span else None
                    ),
                    evidence_tier=EvidenceTier(row.evidence_tier),
                    weight=row.weight,
                    observed_date=date.fromisoformat(row.observed_date)
                    if row.observed_date
                    else None,
                    recency_factor=row.recency_factor,
                    tier_rule=row.tier_rule,
                )
                for row in rows
            ]

    # --- benchmarks ----------------------------------------------------------
    def save_benchmark(self, benchmark: Benchmark) -> None:
        record_id = f"{benchmark.niche}::{benchmark.version}"
        with self._session_factory() as session:
            existing = session.get(BenchmarkRecord, record_id)
            if existing is not None:
                session.delete(existing)
                session.flush()
            session.add(
                BenchmarkRecord(
                    id=record_id,
                    niche=benchmark.niche,
                    version=benchmark.version,
                    required_terms=list(benchmark.required_terms),
                    benchmark_topics=list(benchmark.benchmark_topics),
                    title_formula=benchmark.title_formula,
                    overview_words_min=benchmark.overview_words_min,
                    overview_words_max=benchmark.overview_words_max,
                    portfolio_min_items=benchmark.portfolio_min_items,
                    portfolio_min_quantified=benchmark.portfolio_min_quantified,
                    rate_band=benchmark.rate_band.model_dump(mode="json"),
                    dimension_targets=dict(benchmark.dimension_targets),
                )
            )
            session.commit()

    def get_benchmark(self, niche: str, version: str) -> Benchmark | None:
        record_id = f"{niche}::{version}"
        with self._session_factory() as session:
            row = session.get(BenchmarkRecord, record_id)
            if row is None:
                return None
            return Benchmark(
                niche=row.niche,
                version=row.version,
                required_terms=row.required_terms,
                benchmark_topics=row.benchmark_topics,
                title_formula=row.title_formula,
                overview_words_min=row.overview_words_min,
                overview_words_max=row.overview_words_max,
                portfolio_min_items=row.portfolio_min_items,
                portfolio_min_quantified=row.portfolio_min_quantified,
                rate_band=RateBand.model_validate(row.rate_band),
                dimension_targets=row.dimension_targets,
            )


# The single instance the app imports - mirrors app/storage/store.py's
# `file_store` singleton. Swaps to a different session_factory only in tests.
repository: Repository = SqlAlchemyRepository()
