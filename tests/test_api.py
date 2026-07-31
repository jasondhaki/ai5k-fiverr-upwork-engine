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

import json

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


def test_analyze_persists_claims_and_result_under_the_same_run_id(monkeypatch):
    """
    The whole point of wiring save_claims alongside save_result is that a
    stored Result can be traced back to the EXACT claims that produced it,
    not just "some claims exist somewhere". Proves that end to end through
    a real /analyze request: the run_id on the response header, the Result,
    and the linked Claims all agree, and /report/{run_id} returns both
    together.
    """
    test_repo = _sqlite_repository()
    monkeypatch.setattr("app.platform.api.repository", test_repo)

    ingestion_body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "cut costs by 40 percent",
                    "skill_ids": ["automation"],
                    "evidence_quote": "cut costs by 40 percent for a client",
                }
            ]
        }
    )

    class _FakeIngestClient:
        def complete(self, *, system: str, prompt: str) -> str:
            return ingestion_body

    class _FakeGenClient:
        def complete(self, *, system: str, prompt: str) -> str:
            # No numeric-outcome/proof-tier evidence is required for this
            # test - title/overview drafting just needs to not make a real
            # network call, so both return "nothing to draft".
            return json.dumps({"text": None})

    monkeypatch.setattr(
        "app.ingestion.upwork_parser.build_default_client", lambda: _FakeIngestClient()
    )
    monkeypatch.setattr("app.generation.build_default_client", lambda: _FakeGenClient())

    client = TestClient(app)
    response = client.post(
        "/analyze",
        data={
            "niche": "SMB workflow automation",
            "upwork_text": "cut costs by 40 percent for a client this year",
        },
    )
    assert response.status_code == 200
    run_id = response.headers["X-Run-Id"]

    report = test_repo.get_report(run_id)
    assert report is not None
    result, claims = report
    assert result.run_id == run_id
    assert len(claims) == 1
    assert claims[0].claim_text == "cut costs by 40 percent"

    # Reachable through the API too, not just the repository directly.
    report_response = client.get(f"/report/{run_id}")
    assert report_response.status_code == 200
    body = report_response.json()
    assert body["result"]["run_id"] == run_id
    assert len(body["claims"]) == 1
    assert body["claims"][0]["claim_text"] == "cut costs by 40 percent"


def test_report_endpoint_404s_for_an_unknown_run_id(monkeypatch):
    test_repo = _sqlite_repository()
    monkeypatch.setattr("app.platform.api.repository", test_repo)

    client = TestClient(app)
    response = client.get("/report/does-not-exist")

    assert response.status_code == 404
