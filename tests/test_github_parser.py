"""
GitHub parser tests. The REST call is faked via a session double so these run
offline and deterministically. Real behavior under test: forks are filtered
out, a missing user fails loudly, and claims ground against the repo's own
name/description/README text - never anything invented.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.ingestion.github_parser import (
    DEFAULT_REPO_CAP,
    GitHubFetchError,
    _commit_recency,
    _repo_analysis_cap,
    fetch_repos,
    parse_github,
)
from app.schemas import SourceType
from app.storage.store import file_store

REPOS_PAYLOAD = [
    {
        "name": "rag-pipeline",
        "description": "Built a RAG system over 100k documents",
        "fork": False,
        "url": "https://api.github.com/repos/octocat/rag-pipeline",
        "pushed_at": "2026-07-20T10:15:23Z",
    },
    {
        "name": "forked-thing",
        "description": "A fork, should be filtered out",
        "fork": True,
        "url": "https://api.github.com/repos/octocat/forked-thing",
        "pushed_at": "2026-07-20T10:15:23Z",
    },
]


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        return self._payload


class _FakeSession:
    """Matches routes by substring so tests don't need exact URL construction."""

    def __init__(self, routes: dict[str, _FakeResponse]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        for pattern, response in self.routes.items():
            if pattern in url:
                return response
        return _FakeResponse(404, text="not found")


class _FakeLLMClient:
    def __init__(self, body: str) -> None:
        self.body = body

    def complete(self, *, system: str, prompt: str) -> str:
        return self.body


def test_fetch_repos_filters_forks():
    session = _FakeSession({"/users/octocat/repos": _FakeResponse(200, payload=REPOS_PAYLOAD)})
    repos = fetch_repos("octocat", session=session)
    assert [r["name"] for r in repos] == ["rag-pipeline"]


def test_fetch_repos_404_fails_loudly():
    session = _FakeSession({"/users/ghost/repos": _FakeResponse(404, text="Not Found")})
    with pytest.raises(GitHubFetchError):
        fetch_repos("ghost", session=session)


def test_fetch_repos_non_200_fails_loudly():
    session = _FakeSession({"/users/octocat/repos": _FakeResponse(503, text="rate limited")})
    with pytest.raises(GitHubFetchError):
        fetch_repos("octocat", session=session)


def test_fetch_repos_401_gives_a_generic_auth_failed_message():
    session = _FakeSession(
        {"/users/octocat/repos": _FakeResponse(401, text='{"message": "Bad credentials"}')}
    )
    with pytest.raises(GitHubFetchError, match="GitHub auth failed"):
        fetch_repos("octocat", session=session)


def test_fetch_repos_403_gives_a_rate_limit_message():
    session = _FakeSession({"/users/octocat/repos": _FakeResponse(403, text="rate limited")})
    with pytest.raises(GitHubFetchError, match="GitHub rate limit exceeded"):
        fetch_repos("octocat", session=session)


def test_fetch_repos_429_also_gives_a_rate_limit_message():
    session = _FakeSession({"/users/octocat/repos": _FakeResponse(429, text="too many requests")})
    with pytest.raises(GitHubFetchError, match="GitHub rate limit exceeded"):
        fetch_repos("octocat", session=session)


@pytest.mark.parametrize("status_code", [401, 403, 429, 503])
def test_no_github_fetch_error_ever_echoes_the_token(monkeypatch, status_code):
    secret_token = "ghp_totally_secret_token_value_12345"
    monkeypatch.setenv("GITHUB_TOKEN", secret_token)
    session = _FakeSession(
        {"/users/octocat/repos": _FakeResponse(status_code, text="whatever GitHub sent back")}
    )
    with pytest.raises(GitHubFetchError) as exc_info:
        fetch_repos("octocat", session=session)
    assert secret_token not in str(exc_info.value)


def test_parse_github_grounds_claims_against_repo_text():
    readme_text = "Cut retrieval latency 40% by rewriting the chunking strategy."
    session = _FakeSession(
        {
            "/users/octocat/repos": _FakeResponse(200, payload=[REPOS_PAYLOAD[0]]),
            "/repos/octocat/rag-pipeline/readme": _FakeResponse(200, text=readme_text),
        }
    )
    combined_text = "\n\n".join(["rag-pipeline", REPOS_PAYLOAD[0]["description"], readme_text])
    body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "cut latency 40%",
                    "skill_ids": ["rag_systems"],
                    "evidence_quote": "Cut retrieval latency 40%",
                }
            ]
        }
    )

    claims = parse_github("octocat", session=session, client=_FakeLLMClient(body))

    published = [c for c in claims if c.publishable]
    assert published
    assert published[0].source_type == SourceType.GITHUB_REPO
    assert published[0].source_span.text == "Cut retrieval latency 40%"

    # The span's document_id must resolve to the combined text it was
    # actually grounded against, and link back to the retained raw API data
    # (the repo entry + README) that text was derived from.
    text_id = published[0].source_span.document_id
    assert file_store.get_text(text_id) == combined_text
    original_id = file_store.get_original_id(text_id)
    original = json.loads(file_store.get_bytes(original_id))
    assert original["repo"]["name"] == "rag-pipeline"
    assert original["readme"] == readme_text


def test_commit_recency_parses_pushed_at():
    assert _commit_recency({"pushed_at": "2026-07-20T10:15:23Z"}) == date(2026, 7, 20)


def test_commit_recency_is_none_when_pushed_at_is_missing_or_malformed():
    assert _commit_recency({}) is None
    assert _commit_recency({"pushed_at": "not-a-date"}) is None


def test_parse_github_populates_observed_date_from_last_commit():
    readme_text = "Cut retrieval latency 40% by rewriting the chunking strategy."
    session = _FakeSession(
        {
            "/users/octocat/repos": _FakeResponse(200, payload=[REPOS_PAYLOAD[0]]),
            "/repos/octocat/rag-pipeline/readme": _FakeResponse(200, text=readme_text),
        }
    )
    body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "cut latency 40%",
                    "skill_ids": ["rag_systems"],
                    "evidence_quote": "Cut retrieval latency 40%",
                }
            ]
        }
    )

    claims = parse_github("octocat", session=session, client=_FakeLLMClient(body))

    published = [c for c in claims if c.publishable]
    assert published
    assert published[0].observed_date == date(2026, 7, 20)
    # a repo pushed 9 days ago should barely be discounted
    assert published[0].recency_factor > 0.95


def test_parse_github_discounts_a_stale_repo_via_recency_factor():
    readme_text = "Cut retrieval latency 40% by rewriting the chunking strategy."
    stale_repo = {**REPOS_PAYLOAD[0], "pushed_at": "2019-01-01T00:00:00Z"}
    session = _FakeSession(
        {
            "/users/octocat/repos": _FakeResponse(200, payload=[stale_repo]),
            "/repos/octocat/rag-pipeline/readme": _FakeResponse(200, text=readme_text),
        }
    )
    body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "cut latency 40%",
                    "skill_ids": ["rag_systems"],
                    "evidence_quote": "Cut retrieval latency 40%",
                }
            ]
        }
    )

    claims = parse_github("octocat", session=session, client=_FakeLLMClient(body))

    published = [c for c in claims if c.publishable]
    assert published
    assert published[0].observed_date == date(2019, 1, 1)
    assert published[0].recency_factor < 0.3
    assert published[0].effective_weight < published[0].weight


def test_parse_github_skips_repos_with_no_text():
    session = _FakeSession(
        {
            "/users/octocat/repos": _FakeResponse(
                200,
                payload=[{"name": "", "description": None, "fork": False, "url": None}],
            ),
        }
    )
    claims = parse_github("octocat", session=session, client=_FakeLLMClient('{"claims": []}'))
    assert claims == []


# --- Trivial-README skip: no LLM call spent on near-empty repos -------------


def test_parse_github_skips_a_repo_with_only_a_bare_name_and_no_readme():
    """Below MIN_REPO_TEXT_CHARS - not literally empty (there's a name), but
    nothing substantive enough to spend an extraction call on."""
    tiny_repo = {
        "name": "x",
        "description": None,
        "fork": False,
        "url": "https://api.github.com/repos/octocat/x",
        "pushed_at": "2026-07-20T10:15:23Z",
    }
    session = _FakeSession(
        {
            "/users/octocat/repos": _FakeResponse(200, payload=[tiny_repo]),
            "/repos/octocat/x/readme": _FakeResponse(404, text="not found"),
        }
    )

    class _FailIfCalledClient:
        def complete(self, *, system: str, prompt: str) -> str:
            raise AssertionError("no LLM call should be made for a trivially small repo")

    claims = parse_github("octocat", session=session, client=_FailIfCalledClient())
    assert claims == []


def test_parse_github_still_extracts_from_a_short_but_real_readme():
    """A genuine short README must still clear the threshold and extract
    normally - the skip is for near-empty repos, not brief ones."""
    readme_text = "Cut retrieval latency 40% by rewriting the chunking strategy end to end."
    session = _FakeSession(
        {
            "/users/octocat/repos": _FakeResponse(200, payload=[REPOS_PAYLOAD[0]]),
            "/repos/octocat/rag-pipeline/readme": _FakeResponse(200, text=readme_text),
        }
    )
    body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "cut latency 40%",
                    "skill_ids": ["rag_systems"],
                    "evidence_quote": "Cut retrieval latency 40%",
                }
            ]
        }
    )
    claims = parse_github("octocat", session=session, client=_FakeLLMClient(body))
    assert any(c.publishable for c in claims)


# --- Repo analysis cap: most-recently-pushed N, configurable ----------------


def _repo_payload(name: str, pushed_at: str) -> dict:
    return {
        "name": name,
        "description": "A project",
        "fork": False,
        "url": f"https://api.github.com/repos/octocat/{name}",
        "pushed_at": pushed_at,
    }


def test_repo_analysis_cap_defaults_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("GITHUB_REPO_CAP", raising=False)
    assert _repo_analysis_cap() == DEFAULT_REPO_CAP


def test_repo_analysis_cap_honors_the_env_var(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO_CAP", "3")
    assert _repo_analysis_cap() == 3


def test_repo_analysis_cap_rejects_a_non_positive_value(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO_CAP", "0")
    with pytest.raises(ValueError, match="GITHUB_REPO_CAP"):
        _repo_analysis_cap()


def test_repo_analysis_cap_rejects_a_non_integer_value(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO_CAP", "not-a-number")
    with pytest.raises(ValueError, match="GITHUB_REPO_CAP"):
        _repo_analysis_cap()


def test_parse_github_only_analyzes_the_n_most_recently_pushed_repos(monkeypatch):
    """fetch_repos already returns repos most-recently-pushed-first, so the
    cap keeping the first N is exactly 'the N most recent'. Five repos,
    capped to 2: only the two with the latest pushed_at get extraction calls."""
    monkeypatch.setenv("GITHUB_REPO_CAP", "2")
    payload = [
        _repo_payload("newest", "2026-07-20T00:00:00Z"),
        _repo_payload("middle", "2026-07-10T00:00:00Z"),
        _repo_payload("oldest", "2026-06-01T00:00:00Z"),
    ]
    routes = {"/users/octocat/repos": _FakeResponse(200, payload=payload)}
    for repo in payload:
        routes[f"/repos/octocat/{repo['name']}/readme"] = _FakeResponse(
            200, text="This repo does something specific and real, cutting costs by 40%."
        )
    session = _FakeSession(routes)

    fetched_readme_repos = set()
    orig_get = session.get

    def _tracking_get(url, params=None, headers=None, timeout=None):
        if "/readme" in url:
            fetched_readme_repos.add(url)
        return orig_get(url, params=params, headers=headers, timeout=timeout)

    session.get = _tracking_get

    parse_github("octocat", session=session, client=_FakeLLMClient('{"claims": []}'))

    # only the cap's worth of README fetches happened - the rest were never
    # even fetched, let alone sent to the LLM
    assert len(fetched_readme_repos) == 2
    assert any("newest" in url for url in fetched_readme_repos)
    assert any("middle" in url for url in fetched_readme_repos)
    assert not any("oldest" in url for url in fetched_readme_repos)


def test_parse_github_reports_skipped_by_cap_via_on_status(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO_CAP", "1")
    payload = [
        _repo_payload("newest", "2026-07-20T00:00:00Z"),
        _repo_payload("oldest", "2026-06-01T00:00:00Z"),
    ]
    routes = {"/users/octocat/repos": _FakeResponse(200, payload=payload)}
    for repo in payload:
        routes[f"/repos/octocat/{repo['name']}/readme"] = _FakeResponse(200, text="Something real here.")
    session = _FakeSession(routes)

    events: list[dict] = []
    parse_github(
        "octocat",
        session=session,
        client=_FakeLLMClient('{"claims": []}'),
        on_status=lambda **fields: events.append(fields),
    )

    assert any("skipped by cap" in (e.get("detail") or "") for e in events)
