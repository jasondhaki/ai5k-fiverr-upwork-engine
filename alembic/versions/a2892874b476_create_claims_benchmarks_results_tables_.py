"""create claims benchmarks results tables and pgvector extension

Revision ID: a2892874b476
Revises:
Create Date: 2026-07-31 03:45:04.489246

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from app.storage.models import BENCHMARK_TOPIC_EMBEDDING_DIM, JSONB_OR_JSON

# revision identifiers, used by Alembic.
revision: str = 'a2892874b476'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "benchmarks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("niche", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("required_terms", JSONB_OR_JSON, nullable=False),
        sa.Column("benchmark_topics", JSONB_OR_JSON, nullable=False),
        # Ready for semantic-coverage embeddings later (spec section 5) -
        # column + pgvector extension only, nothing writes to it yet.
        sa.Column(
            "benchmark_topics_embedding",
            Vector(BENCHMARK_TOPIC_EMBEDDING_DIM).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
        ),
        sa.Column("title_formula", sa.String(), nullable=False),
        sa.Column("overview_words_min", sa.Integer(), nullable=False),
        sa.Column("overview_words_max", sa.Integer(), nullable=False),
        sa.Column("portfolio_min_items", sa.Integer(), nullable=False),
        sa.Column("portfolio_min_quantified", sa.Integer(), nullable=False),
        sa.Column("rate_band", JSONB_OR_JSON, nullable=False),
        sa.Column("dimension_targets", JSONB_OR_JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_benchmarks_niche", "benchmarks", ["niche"])

    op.create_table(
        "results",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("niche", sa.String(), nullable=False),
        sa.Column("benchmark_version", sa.String(), nullable=False),
        sa.Column("readiness", sa.Float(), nullable=False),
        sa.Column("capped", sa.Boolean(), nullable=False),
        sa.Column("cap_note", sa.String(), nullable=True),
        sa.Column("dimensions", JSONB_OR_JSON, nullable=False),
        sa.Column("blocking", JSONB_OR_JSON, nullable=False),
        sa.Column("gaps", JSONB_OR_JSON, nullable=False),
        sa.Column("generated", JSONB_OR_JSON, nullable=False),
        sa.Column("generation_incomplete", sa.Boolean(), nullable=False),
        sa.Column("total_claims", sa.Integer(), nullable=False),
        sa.Column("provable_claims", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "claims",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("result_id", sa.String(), nullable=True),
        sa.Column("claim_text", sa.String(), nullable=False),
        sa.Column("skill_ids", JSONB_OR_JSON, nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("source_span", JSONB_OR_JSON, nullable=True),
        sa.Column("evidence_tier", sa.String(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("observed_date", sa.String(), nullable=True),
        sa.Column("recency_factor", sa.Float(), nullable=False),
        sa.Column("tier_rule", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_claims_result_id", "claims", ["result_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_claims_result_id", table_name="claims")
    op.drop_table("claims")
    op.drop_table("results")
    op.drop_index("ix_benchmarks_niche", table_name="benchmarks")
    op.drop_table("benchmarks")
    # Deliberately NOT dropping the vector extension - it may be relied on by
    # other tables/migrations added later, and dropping a Postgres extension
    # is a shared, database-wide action rather than something scoped to what
    # this migration created.
