"""
GitHub parser: pulls a user's public, non-fork repos via the REST API and
turns each repo's name/description/README into grounded claims.

Read-only, unauthenticated calls work but are rate-limited to 60/hour; set
GITHUB_TOKEN in the environment to raise that limit. No GraphQL needed for
this slice - REST's per-repo README endpoint is enough.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Callable

import requests

from app.ingestion.extractor import LLMClient, build_default_client, extract_candidate_claims
from app.schemas import Claim, SourceType
from app.storage.store import file_store

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
REQUEST_TIMEOUT = 10

# Below this many characters of combined name+description+README, there's
# nothing substantive enough to justify an extraction call - e.g. a bare repo
# name and a one-line description with no README at all. Deliberately small
# (a genuine short README still clears it easily): the goal is to catch
# near-empty repos, not to second-guess real-but-brief documentation.
MIN_REPO_TEXT_CHARS = 40

# How many of the user's most-recently-pushed repos actually get analyzed
# (README fetch + LLM extraction) - the expensive part. fetch_repos already
# returns repos sorted pushed-first (GitHub API's own sort=pushed), so
# slicing the first N here is exactly "the N most recently pushed repos".
# Deliberately not unlimited: with recency_factor (config/weights.py)
# already discounting old evidence heavily, a stale 20th repo contributes
# little to the score but costs the same LLM calls as a fresh one -
# GITHUB_REPO_CAP overrides this default, never a code change.
DEFAULT_REPO_CAP = 12


class GitHubFetchError(ValueError):
    """Raised when the GitHub API can't be reached or the user doesn't exist.
    Never swallowed - a typo'd username should not silently look like "no repos"."""


def _repo_analysis_cap() -> int:
    """GITHUB_REPO_CAP overrides DEFAULT_REPO_CAP; unset/blank keeps the
    default. Fails loudly on a non-positive-integer value rather than
    silently falling back, matching how every other env-var-selected knob
    in this codebase behaves (see CLAUDE.md's "Swappable backends")."""
    raw = os.environ.get("GITHUB_REPO_CAP", "").strip()
    if not raw:
        return DEFAULT_REPO_CAP
    try:
        cap = int(raw)
    except ValueError as exc:
        raise ValueError(f"GITHUB_REPO_CAP must be an integer, got {raw!r}") from exc
    if cap <= 0:
        raise ValueError(f"GITHUB_REPO_CAP must be a positive integer, got {cap}")
    return cap


def _headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    headers = {"Accept": accept}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_repos(username: str, session: requests.Session | None = None) -> list[dict]:
    """Fetch public, non-fork repos for `username`, most-recently-pushed first."""
    http = session or requests
    response = http.get(
        f"{GITHUB_API}/users/{username}/repos",
        params={"sort": "pushed", "per_page": 20, "type": "owner"},
        headers=_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 404:
        raise GitHubFetchError(f"GitHub user '{username}' does not exist.")
    if response.status_code == 401:
        # Never echo GITHUB_TOKEN or the response body back - a bad/expired
        # token gets a fixed, generic message, not whatever GitHub sent back.
        raise GitHubFetchError("GitHub auth failed - check GITHUB_TOKEN.")
    if response.status_code in (403, 429):
        raise GitHubFetchError(
            "GitHub rate limit exceeded - set GITHUB_TOKEN to raise the "
            "limit, or wait and retry."
        )
    if response.status_code != 200:
        raise GitHubFetchError(
            f"GitHub API returned {response.status_code} for user '{username}'."
        )
    repos = response.json()
    return [repo for repo in repos if not repo.get("fork")]


def _fetch_readme(repo: dict, session: requests.Session | None = None) -> str:
    repo_url = repo.get("url")
    if not repo_url:
        return ""
    http = session or requests
    try:
        response = http.get(
            f"{repo_url}/readme",
            headers=_headers(accept="application/vnd.github.raw+json"),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        return ""  # README is best-effort; name/description alone still ground
    return response.text if response.status_code == 200 else ""


def _commit_recency(repo: dict) -> date | None:
    """
    The repo list endpoint already returns `pushed_at` - the timestamp of the
    last push to the default branch, i.e. the last commit - so this needs no
    extra API call. Malformed or missing timestamps discount to None (treated
    as "unknown age", never as "brand new") rather than raising, since a
    parse hiccup here shouldn't fail the whole repo.
    """
    raw = repo.get("pushed_at")
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").date()
    except ValueError:
        return None


def _repo_text(repo: dict, readme: str) -> str:
    parts = [repo.get("name") or "", repo.get("description") or ""]
    if readme:
        parts.append(readme)
    return "\n\n".join(part for part in parts if part)


def parse_github(
    username: str,
    session: requests.Session | None = None,
    client: LLMClient | None = None,
    on_status: Callable[..., None] | None = None,
) -> list[Claim]:
    """Pull repos for `username` and extract grounded claims from each repo's
    name + description + README text."""
    if on_status is not None:
        on_status(stage="fetching_repos", detail=f"Fetching repos for {username}")
    repos = fetch_repos(username, session=session)
    logger.info("Found %d non-fork repos for %s", len(repos), username)

    # Already sorted most-recently-pushed-first (fetch_repos requests
    # sort=pushed), so slicing here IS "keep the N most recent" - no
    # re-sorting needed.
    cap = _repo_analysis_cap()
    total_found = len(repos)
    skipped_by_cap = repos[cap:]
    repos = repos[:cap]
    if skipped_by_cap:
        logger.info(
            "Capping analysis to the %d most recently pushed repos for %s (%d skipped by cap)",
            cap, username, len(skipped_by_cap),
        )

    if on_status is not None:
        detail = f"Found {total_found} non-fork repositories"
        if skipped_by_cap:
            detail += (
                f" - analyzing the {len(repos)} most recently pushed "
                f"({len(skipped_by_cap)} skipped by cap)"
            )
        on_status(detail=detail)
    llm_client = client or build_default_client()

    claims: list[Claim] = []
    for index, repo in enumerate(repos, start=1):
        repo_name = repo.get("name") or username
        if on_status is not None:
            on_status(
                stage="fetching_repos",
                detail=f"repo {index}/{len(repos)}: {repo_name}",
                progress_current=index,
                progress_total=len(repos),
            )
        readme = _fetch_readme(repo, session=session)
        text = _repo_text(repo, readme)
        if len(text.strip()) < MIN_REPO_TEXT_CHARS:
            logger.info(
                "[%d/%d] repo %s has too little text (%d chars < %d), skipping - no LLM call spent",
                index, len(repos), repo_name, len(text.strip()), MIN_REPO_TEXT_CHARS,
            )
            if on_status is not None:
                on_status(detail=f"Skipped {repo_name} (no README)")
            continue
        logger.info("[%d/%d] Starting extraction for repo %s", index, len(repos), repo_name)
        if on_status is not None:
            on_status(
                stage="extracting_claims",
                detail=f"repo {index}/{len(repos)}: {repo_name}",
                progress_current=index,
                progress_total=len(repos),
            )
        # The original is the raw API data (the repo list entry plus its
        # README) - retained as-is, distinct from `text`, the derived string
        # spans are actually grounded against.
        original = json.dumps({"repo": repo, "readme": readme}, indent=2).encode("utf-8")
        pair = file_store.put_source(original=original, text=text, original_suffix=".json")
        repo_claims = extract_candidate_claims(
            llm_client,
            document_id=pair.text_id,
            text=text,
            source_type=SourceType.GITHUB_REPO,
            locator_prefix=f"repo {repo_name}",
            observed_date=_commit_recency(repo),
        )
        claims.extend(repo_claims)
        # This is a local, per-repo delta for the activity log only - NOT the
        # running claims_found/claims_provable total (route_input, which sees
        # every source, is still the only place that reports that; see below).
        if on_status is not None:
            if repo_claims:
                repo_provable = sum(1 for c in repo_claims if c.publishable)
                on_status(
                    detail=f"+{len(repo_claims)} claim"
                    f"{'' if len(repo_claims) == 1 else 's'} found ({repo_provable} provable)"
                )
            else:
                on_status(detail=f"No claims found in {repo_name}")
        # claims_found/claims_provable are NOT reported per-repo here: this
        # loop only ever sees GitHub's own claims, not the running total
        # across CV/Upwork too - route_input (which sees the full picture)
        # is the only place that reports those, once this whole call returns.
    return claims
