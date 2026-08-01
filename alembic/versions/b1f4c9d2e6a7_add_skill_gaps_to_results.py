"""add skill_gaps to results

Revision ID: b1f4c9d2e6a7
Revises: a2892874b476
Create Date: 2026-08-01 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from app.storage.models import JSONB_OR_JSON

# revision identifiers, used by Alembic.
revision: str = 'b1f4c9d2e6a7'
down_revision: Union[str, Sequence[str], None] = 'a2892874b476'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable, no server_default: existing rows get NULL rather than
    # requiring a backfill value, and ResultRecord/get_result already treat
    # NULL identically to an empty list (see that column's own comment) -
    # this stays a plain ADD COLUMN, not a rewrite of every existing row.
    op.add_column("results", sa.Column("skill_gaps", JSONB_OR_JSON, nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("results", "skill_gaps")
