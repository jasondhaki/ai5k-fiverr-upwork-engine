"""
Proves the async /analyze flow works end to end through the Repository
interface and the status store - not just that pages render (that's
test_pipeline_skeleton.py's job for the synchronous pipeline itself).

Monkeypatches app.platform.api.repository to a SQLite-backed repository
(same pattern as monkeypatching build_default_client elsewhere) so this
never touches the real DATABASE_URL-backed singleton or the network, even
though a real Neon URL is present in .env.

Note on timing: FastAPI's TestClient drives BackgroundTasks to completion
before .post(...) returns control to the caller, so in practice every poll
loop below resolves on its first iteration. The tests still poll in a loop
(rather than assuming that), because that's the real contract
/analyze/{run_id}/status promises regardless of this test client's
particular execution model.
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.ingestion.github_parser import GitHubFetchError
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


def _run_id_from_redirect(response) -> str:
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/analyze/")
    return location.rsplit("/", 1)[-1]


def _poll_until_terminal(client: TestClient, run_id: str, max_polls: int = 50) -> dict:
    for _ in range(max_polls):
        response = client.get(f"/analyze/{run_id}/status")
        assert response.status_code == 200
        status = response.json()
        if status["stage"] in ("done", "error"):
            return status
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} never reached a terminal stage after {max_polls} polls")


# --- POST /analyze: redirects, never renders the report inline any more -----


def test_analyze_redirects_to_the_progress_page_instead_of_rendering_inline(monkeypatch):
    test_repo = _sqlite_repository()
    monkeypatch.setattr("app.platform.api.repository", test_repo)

    client = TestClient(app)
    response = client.post(
        "/analyze", data={"niche": "SMB workflow automation"}, follow_redirects=False
    )

    run_id = _run_id_from_redirect(response)
    assert run_id  # non-empty - a real run_id was minted and put in the URL


def test_progress_page_renders_for_a_run_the_post_just_created(monkeypatch):
    test_repo = _sqlite_repository()
    monkeypatch.setattr("app.platform.api.repository", test_repo)

    client = TestClient(app)
    post_response = client.post(
        "/analyze", data={"niche": "SMB workflow automation"}, follow_redirects=False
    )
    run_id = _run_id_from_redirect(post_response)

    progress_response = client.get(f"/analyze/{run_id}")

    assert progress_response.status_code == 200
    assert run_id in progress_response.text  # the JS needs it to poll the right URL


# --- Persistence: the whole point of wiring save_claims alongside save_result -


def test_analyze_persists_the_result_via_the_repository(monkeypatch):
    test_repo = _sqlite_repository()
    monkeypatch.setattr("app.platform.api.repository", test_repo)

    client = TestClient(app)
    post_response = client.post(
        "/analyze", data={"niche": "SMB workflow automation"}, follow_redirects=False
    )
    run_id = _run_id_from_redirect(post_response)

    status = _poll_until_terminal(client, run_id)
    assert status["stage"] == "done"

    with test_repo._session_factory() as session:
        from app.storage.models import ResultRecord

        rows = session.query(ResultRecord).all()

    assert len(rows) == 1
    assert rows[0].niche == "SMB workflow automation"


def test_analyze_persists_claims_and_result_under_the_same_run_id(monkeypatch):
    """
    A stored Result can be traced back to the EXACT claims that produced it,
    not just "some claims exist somewhere". Proves that end to end through a
    real (background-task-driven) /analyze request: the run_id in the
    redirect, the polled status, the Result, and the linked Claims all
    agree, and both /report/{run_id} (JSON) and /analyze/{run_id}/result
    (HTML) return the same finished report.
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
    post_response = client.post(
        "/analyze",
        data={
            "niche": "SMB workflow automation",
            "upwork_text": "cut costs by 40 percent for a client this year",
        },
        follow_redirects=False,
    )
    run_id = _run_id_from_redirect(post_response)

    status = _poll_until_terminal(client, run_id)
    assert status["stage"] == "done"
    assert status["claims_found"] == 1
    assert status["claims_provable"] == 1

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

    # And the same HTML report page /analyze used to render inline before -
    # now only reachable once the background run has actually finished.
    result_page = client.get(f"/analyze/{run_id}/result")
    assert result_page.status_code == 200
    # Checks the underlying data made it onto the page, not exact markup -
    # matches the claims_found/claims_provable counts asserted above.
    assert "can prove 1" in result_page.text
    # Benchmark data pulled in for display (not stored on Result itself) -
    # the pricing dimension's rate band and at least one dimension's target.
    # Labeled as benchmark context, not a conclusion about this user's own
    # evidence (see the pricing_strategy Part-6 fix) - the number is the
    # same, the framing around it changed.
    assert "Top tier in this niche charges $50" in result_page.text
    assert "not a conclusion about your own evidence" in result_page.text
    assert "Target:" in result_page.text
    # Four of the seven dimensions are heuristic estimates over raw evidence
    # text, not direct measurements (see is_provisional) - the report must
    # say so visibly, not present all seven as equally measured.
    assert "Estimated" in result_page.text
    assert "measured directly" in result_page.text
    # This profile's one claim is grounded (provable), so blocking is empty -
    # that must read as "here's specifically what was checked", never as an
    # unqualified all-clear on things (ToS risk, identity) with no signal at all.
    assert "No unproven claims found" in result_page.text
    assert "Terms-of-service risk and identity verification aren" in result_page.text
    # None of "cut costs by 40 percent" matches any required term/topic in
    # the real SMB-automation benchmark, so the skill-gap section - a report,
    # never a scoring dimension - should list at least one of them.
    assert "What to build next" in result_page.text
    assert "n8n" in result_page.text
    # The audit-trail link, pointing at the same run_id.
    assert f"/analyze/{run_id}/claims" in result_page.text

    # The claim audit trail itself: same run_id, real claim/tier/span data.
    claims_page = client.get(f"/analyze/{run_id}/claims")
    assert claims_page.status_code == 200
    assert "cut costs by 40 percent" in claims_page.text
    assert "T" in claims_page.text  # some evidence tier tag renders
    assert "cut costs by 40 percent for a client" in claims_page.text  # the span text


# --- /analyze/{run_id}/claims: the claim-level audit trail -------------------


def test_unproven_claims_are_persisted_and_rendered_not_filtered_out(monkeypatch):
    """
    A claim whose evidence_quote doesn't ground against the source still
    gets extracted, still gets persisted via save_claims, and must still
    show up - distinctly marked - on both the claims audit trail and the
    report page's "fix before publishing" section. Nothing in the Repository
    or these two routes may filter it out by publishable status.
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
                },
                {
                    "claim_text": "led a team of five engineers",
                    "skill_ids": ["leadership"],
                    "evidence_quote": "this exact phrase never appears anywhere in the source text",
                },
            ]
        }
    )

    class _FakeIngestClient:
        def complete(self, *, system: str, prompt: str) -> str:
            return ingestion_body

    class _FakeGenClient:
        def complete(self, *, system: str, prompt: str) -> str:
            return json.dumps({"text": None})

    monkeypatch.setattr(
        "app.ingestion.upwork_parser.build_default_client", lambda: _FakeIngestClient()
    )
    monkeypatch.setattr("app.generation.build_default_client", lambda: _FakeGenClient())

    client = TestClient(app)
    post_response = client.post(
        "/analyze",
        data={
            "niche": "SMB workflow automation",
            "upwork_text": "cut costs by 40 percent for a client this year",
        },
        follow_redirects=False,
    )
    run_id = _run_id_from_redirect(post_response)
    status = _poll_until_terminal(client, run_id)
    assert status["stage"] == "done"

    # The repository itself never filters by publishable status.
    claims = test_repo.get_claims(run_id)
    assert len(claims) == 2
    assert sum(1 for c in claims if c.publishable) == 1
    assert sum(1 for c in claims if not c.publishable) == 1

    # The audit trail shows both, with the ungrounded one clearly marked and
    # the source-group header split into proven/unproven counts.
    claims_page = client.get(f"/analyze/{run_id}/claims").text
    assert "led a team of five engineers" in claims_page
    assert "NOT PROVEN" in claims_page
    assert "no matching text found in source" in claims_page
    assert "1 proven, 1 unproven" in claims_page

    # The report page itemizes the specific unproven claim text, not just a
    # bare "1 of 2 claims unproven" count.
    result_page = client.get(f"/analyze/{run_id}/result").text
    assert "1 of 2 claims unproven" in result_page  # the existing summary line stays
    assert "led a team of five engineers" in result_page  # the itemized claim text
    assert "NOT PROVEN" in result_page


def test_claims_page_404s_for_an_unknown_run_id(monkeypatch):
    test_repo = _sqlite_repository()
    monkeypatch.setattr("app.platform.api.repository", test_repo)

    client = TestClient(app)
    assert client.get("/analyze/does-not-exist/claims").status_code == 404


# --- /report/{run_id}: unchanged JSON contract -------------------------------


def test_report_endpoint_404s_for_an_unknown_run_id(monkeypatch):
    test_repo = _sqlite_repository()
    monkeypatch.setattr("app.platform.api.repository", test_repo)

    client = TestClient(app)
    response = client.get("/report/does-not-exist")

    assert response.status_code == 404


# --- Unknown run_id: every /analyze/{run_id}* route 404s, never 200s with junk --


def test_status_endpoint_404s_for_an_unknown_run_id():
    client = TestClient(app)
    assert client.get("/analyze/does-not-exist/status").status_code == 404


def test_progress_page_404s_for_an_unknown_run_id():
    client = TestClient(app)
    assert client.get("/analyze/does-not-exist").status_code == 404


def test_result_page_404s_for_an_unknown_run_id(monkeypatch):
    test_repo = _sqlite_repository()
    monkeypatch.setattr("app.platform.api.repository", test_repo)

    client = TestClient(app)
    assert client.get("/analyze/does-not-exist/result").status_code == 404


# --- Status endpoint: in-progress and error snapshots ------------------------


def test_status_endpoint_reports_an_in_progress_stage_with_progress_fields():
    """
    Directly exercises the status store + endpoint together: a real pipeline
    run completes synchronously under TestClient's background-task execution
    model (see module docstring), so this is the only way to observe an
    in-progress snapshot through the HTTP layer rather than the already-
    finished one - proves the endpoint faithfully reports whatever the store
    currently holds, including the per-repo progress fields.
    """
    from app.platform.status import status_store

    run_id = "test-in-progress-" + uuid.uuid4().hex
    status_store.create(run_id)
    status_store.update(
        run_id,
        stage="fetching_repos",
        detail="repo 2/5: digital-workshop",
        progress_current=2,
        progress_total=5,
        claims_found=3,
        claims_provable=1,
    )

    client = TestClient(app)
    response = client.get(f"/analyze/{run_id}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["stage"] == "fetching_repos"
    assert body["detail"] == "repo 2/5: digital-workshop"
    assert body["progress_current"] == 2
    assert body["progress_total"] == 5
    assert body["claims_found"] == 3
    assert body["claims_provable"] == 1
    # Not currently rate-limited - the endpoint reports that explicitly
    # rather than omitting the key, so the progress page never has to treat
    # "absent" and "zero seconds left" as the same thing.
    assert body["rate_limit_seconds_remaining"] is None
    # The raw absolute timestamp is an internal detail for computing that
    # field fresh on every poll - never handed to the browser directly.
    assert "rate_limit_resume_at" not in body


def test_status_endpoint_reports_seconds_remaining_computed_fresh_from_the_server_clock():
    """
    RunStatus stores an absolute rate_limit_resume_at (see status.py's
    docstring on that field for why: an absolute server-clock timestamp,
    never something the browser's own clock has to agree with). The status
    endpoint's job is to turn that into "how many seconds are left, right
    now" on every single call - this proves that conversion, and that it
    floors at zero rather than ever going negative once the deadline has
    passed.
    """
    from app.platform.status import status_store

    run_id = "test-rate-limited-" + uuid.uuid4().hex
    status_store.create(run_id)
    status_store.update(
        run_id,
        stage="extracting_claims",
        detail="Rate-limited by the AI provider - waiting to retry",
        rate_limit_resume_at=time.time() + 30,
    )

    client = TestClient(app)
    body = client.get(f"/analyze/{run_id}/status").json()

    assert body["rate_limit_seconds_remaining"] is not None
    assert 0 < body["rate_limit_seconds_remaining"] <= 30

    # Once the deadline is in the past, it reports exactly 0 - never negative.
    status_store.update(run_id, rate_limit_resume_at=time.time() - 5)
    body = client.get(f"/analyze/{run_id}/status").json()
    assert body["rate_limit_seconds_remaining"] == 0.0


def test_status_endpoint_reports_done_with_final_claim_counts(monkeypatch):
    test_repo = _sqlite_repository()
    monkeypatch.setattr("app.platform.api.repository", test_repo)

    client = TestClient(app)
    post_response = client.post(
        "/analyze", data={"niche": "SMB workflow automation"}, follow_redirects=False
    )
    run_id = _run_id_from_redirect(post_response)

    status = _poll_until_terminal(client, run_id)

    assert status["stage"] == "done"
    assert status["error"] is None


def test_status_endpoint_reports_error_with_a_message_on_a_real_failure(monkeypatch):
    """
    A real failure, not a synthetic status-store poke: monkeypatches the
    GitHub fetch to raise, proving the background task's own exception
    handling (app.platform.api._run_pipeline_in_background) actually
    surfaces the failure through the status store instead of the exception
    vanishing into BackgroundTasks with nothing to show the person watching
    the progress page.
    """
    test_repo = _sqlite_repository()
    monkeypatch.setattr("app.platform.api.repository", test_repo)

    def _raise(*args, **kwargs):
        raise GitHubFetchError("GitHub user 'this-should-not-exist' does not exist.")

    monkeypatch.setattr("app.ingestion.github_parser.fetch_repos", _raise)

    client = TestClient(app)
    post_response = client.post(
        "/analyze",
        data={"niche": "SMB workflow automation", "github_username": "this-should-not-exist"},
        follow_redirects=False,
    )
    run_id = _run_id_from_redirect(post_response)

    status = _poll_until_terminal(client, run_id)

    assert status["stage"] == "error"
    assert "does not exist" in status["error"]
