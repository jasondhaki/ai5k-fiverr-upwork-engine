"""
Proves /analyze actually persists the Result it renders, through the
Repository interface - not just that the page renders (that's
test_pipeline_skeleton.py's job).

Monkeypatches app.platform.api.repository to a SQLite-backed repository
(same pattern as monkeypatching build_default_client elsewhere) so this
never touches the real DATABASE_URL-backed singleton or the network, even
though a real Neon URL is present in .env.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.platform.api import app
from app.storage.models import Base
from app.storage.repository import SqlAlchemyRepository


def _sqlite_repository() -> SqlAlchemyRepository:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return SqlAlchemyRepository(session_factory=session_factory)


def test_analyze_persists_the_result_via_the_repository(monkeypatch):
    test_repo = _sqlite_repository()
    monkeypatch.setattr("app.platform.api.repository", test_repo)

    client = TestClient(app)
    response = client.post("/analyze", data={"niche": "SMB workflow automation"})

    assert response.status_code == 200

    with test_repo._session_factory() as session:
        from app.storage.models import ResultRecord

        rows = session.query(ResultRecord).all()

    assert len(rows) == 1
    assert rows[0].niche == "SMB workflow automation"
