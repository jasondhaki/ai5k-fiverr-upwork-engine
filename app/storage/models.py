"""
SQLAlchemy ORM models for the three frozen record shapes (app/schemas/).

One table per top-level model - claims, benchmarks, results - with nested
structures (source_span, dimensions, gaps, generated, rate_band) kept as JSONB
rather than fully normalized. The frozen Pydantic schemas are still the only
contract the rest of the app depends on; these tables exist purely so a
Repository (app/storage/repository.py) can round-trip them to Postgres. No
code outside app/storage/ should import this module directly - go through the
Repository interface instead, the same way file storage is never touched
except through the FileStore protocol.

JSONB is Postgres-only, so every JSON-ish column uses `.with_variant(JSON(),
"sqlite")` to fall back to plain JSON under the fast test suite's in-memory
SQLite engine (see tests/test_repository.py) while still getting real JSONB
on Neon.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Ready for semantic-coverage embeddings later (spec section 5's keyword
# score, 30% semantic half) - column and pgvector extension only for now,
# nothing computes or writes a real embedding yet. 384 matches the
# commented-out sentence-transformers/all-MiniLM-L6-v2 dependency already
# noted in requirements.txt as the likely local-embedding model.
BENCHMARK_TOPIC_EMBEDDING_DIM = 384

JSONB_OR_JSON = JSONB().with_variant(JSON(), "sqlite")


def _uuid_hex() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class ClaimRecord(Base):
    """Mirrors app/schemas/claim.py's Claim. `id` is synthetic (Claim itself
    has no id field) so a claim can be referenced from a ResultRecord."""

    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid_hex)
    result_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    claim_text: Mapped[str] = mapped_column(String, nullable=False)
    skill_ids: Mapped[list] = mapped_column(JSONB_OR_JSON, nullable=False, default=list)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_span: Mapped[dict | None] = mapped_column(JSONB_OR_JSON, nullable=True)
    evidence_tier: Mapped[str] = mapped_column(String, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    observed_date: Mapped[str | None] = mapped_column(String, nullable=True)
    recency_factor: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    tier_rule: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class BenchmarkRecord(Base):
    """Mirrors app/schemas/benchmark.py's Benchmark. Primary key is the
    (niche, version) pair as one string, since a benchmark is looked up by
    exactly that pair and never "the latest" (see app/evidence/benchmarks.py)."""

    __tablename__ = "benchmarks"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # f"{niche}::{version}"
    niche: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[str] = mapped_column(String, nullable=False)

    required_terms: Mapped[list] = mapped_column(JSONB_OR_JSON, nullable=False, default=list)
    benchmark_topics: Mapped[list] = mapped_column(JSONB_OR_JSON, nullable=False, default=list)
    # Not populated or read by any code yet - see BENCHMARK_TOPIC_EMBEDDING_DIM.
    benchmark_topics_embedding: Mapped[list | None] = mapped_column(
        Vector(BENCHMARK_TOPIC_EMBEDDING_DIM).with_variant(JSON(), "sqlite"), nullable=True
    )

    title_formula: Mapped[str] = mapped_column(String, nullable=False)
    overview_words_min: Mapped[int] = mapped_column(Integer, nullable=False)
    overview_words_max: Mapped[int] = mapped_column(Integer, nullable=False)
    portfolio_min_items: Mapped[int] = mapped_column(Integer, nullable=False)
    portfolio_min_quantified: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    rate_band: Mapped[dict] = mapped_column(JSONB_OR_JSON, nullable=False)
    dimension_targets: Mapped[dict] = mapped_column(JSONB_OR_JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ResultRecord(Base):
    """Mirrors app/schemas/result.py's Result - one row per pipeline run."""

    __tablename__ = "results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid_hex)

    niche: Mapped[str] = mapped_column(String, nullable=False)
    benchmark_version: Mapped[str] = mapped_column(String, nullable=False)

    readiness: Mapped[float] = mapped_column(Float, nullable=False)
    capped: Mapped[bool] = mapped_column(Boolean, nullable=False)
    cap_note: Mapped[str | None] = mapped_column(String, nullable=True)

    dimensions: Mapped[list] = mapped_column(JSONB_OR_JSON, nullable=False)
    blocking: Mapped[list] = mapped_column(JSONB_OR_JSON, nullable=False, default=list)
    gaps: Mapped[list] = mapped_column(JSONB_OR_JSON, nullable=False, default=list)
    generated: Mapped[list] = mapped_column(JSONB_OR_JSON, nullable=False, default=list)
    generation_incomplete: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    total_claims: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    provable_claims: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Nullable (not default=list-backed NOT NULL) specifically so the Alembic
    # migration adding this column never has to backfill a value onto rows
    # that already exist in a real deployed database - see that migration's
    # own comment. get_result treats a NULL row here identically to [].
    skill_gaps: Mapped[list | None] = mapped_column(JSONB_OR_JSON, nullable=True, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
