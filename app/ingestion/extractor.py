"""
The parallel LLM extractor: turns raw source text into candidate claims.

Up to three small, independent passes run per document - identity, history,
skills - each schema-validated and retried once on failure. Which passes
actually run is chosen per source_type (see _PASSES_BY_SOURCE below): a CV
carries role, project history, AND a skills list, so it gets all three, but a
GitHub README has no identity language to find and a pasted Upwork summary
has no separate work-history section beyond what identity/skills already
cover - running a pass that structurally cannot find anything for that source
type only spends an LLM call to relearn what the source_type already implies.
The model never returns claim text to trust blindly: it returns a verbatim
quote copied from the exact
text it was given, and the caller locates that quote via plain string search
and re-slices the literal substring via `ground_span`. The model is NOT asked
to compute character indices - LLMs are reliably bad at that arithmetic even
over a few hundred characters (verified empirically: a 334-character source
produced offsets wrong by dozens of characters, and a 7898-character README
produced offsets wrong by anywhere from -2393 to +3193 with no consistent
pattern). Turning "where is this in the text" into a search problem our own
code solves deterministically, instead of an arithmetic problem the model
solves unreliably, is the fix. A span that fails to ground - because the
quote isn't a verbatim substring, or is too short to trust - still produces a
Claim with `source_span=None`, so it surfaces as a coaching prompt instead of
vanishing or, worse, publishing a paraphrase as fact.

Anthropic SDK directly, no LangChain: the span round-trip is exactly the part
an abstraction layer would hide, and it is the one thing here that must stay
inspectable. Groq is called over its OpenAI-compatible REST endpoint instead
of a second SDK, since the extractor only ever needs one method
(`complete(system, prompt) -> text`) - not worth a new dependency.

Both clients satisfy the same `LLMClient` protocol, so `_run_pass` and
`extract_candidate_claims` never know or care which provider answered a given
pass - the retry-on-malformed-JSON logic and the span-grounding rules below
apply identically regardless of which client `build_default_client()` picked.

Running identity/history/skills independently means the same real
accomplishment is often rediscovered by more than one pass - the skills pass
and the history pass both noticing the same repo highlight, say. Before
returning, `_dedupe_claims` collapses exact-text repeats and near-duplicate
rewordings (via plain string similarity, not another LLM call) down to one
claim each, so a single accomplishment never inflates a profile by appearing
two or three times over.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import difflib
import json
import logging
import os
import re
import time
from datetime import date
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

# Ambient "which source is this pass working on" context for log messages
# only - never read for any decision logic. Set once per
# extract_candidate_claims call and explicitly propagated into the thread
# pool's workers (ThreadPoolExecutor does NOT copy contextvars into worker
# threads on its own - verified empirically), so a retry logged from deep
# inside _run_pass or a provider client's own retry loop can still say which
# CV/repo/paste it was retrying for, without threading a parameter through
# the LLMClient protocol that every client and every test implements.
_source_label: contextvars.ContextVar[str] = contextvars.ContextVar(
    "source_label", default="unknown source"
)

# Same propagation mechanism as _source_label above, for a different purpose:
# lets GroqClient's rate-limit backoff (the retry loop in .complete() below)
# surface a human-readable "waiting Ns (API rate limit)" detail through the
# async /analyze progress page, without extract_candidate_claims or
# generate_title_draft/generate_overview_draft needing an on_status parameter
# of their own - status_reporting() is entered ONCE, by run_pipeline, around
# the whole run (both ingestion's parallel LLM calls AND generation's direct
# ones), and every ThreadPoolExecutor submission below already copies
# whatever this is set to via contextvars.copy_context().run(...).
_status_callback: contextvars.ContextVar[Callable[..., None] | None] = contextvars.ContextVar(
    "status_callback", default=None
)


@contextlib.contextmanager
def status_reporting(on_status: Callable[..., None] | None):
    """Makes `on_status` readable from GroqClient's retry loop (and anywhere
    else in this module) for the duration of the `with` block. A no-op
    context (nothing set, nothing to reset) when `on_status` is None, which
    is the case for every one of the ~230 existing tests that call
    run_pipeline directly without one."""
    if on_status is None:
        yield
        return
    token = _status_callback.set(on_status)
    try:
        yield
    finally:
        _status_callback.reset(token)


import requests
from pydantic import BaseModel, ValidationError

from app.config.weights import TIER_WEIGHTS, recency_factor as compute_recency_factor
from app.ingestion.span_grounding import SpanGroundingError, ground_span
from app.schemas import Claim, SourceSpan, SourceType
from app.schemas.claim import EvidenceTier

RETRY_ATTEMPTS = 2  # one initial try + one retry on a malformed response


class LLMClient(Protocol):
    """The only shape the extractor depends on. Anthropic's client satisfies
    it via `AnthropicClient` below; tests supply a fake."""

    def complete(self, *, system: str, prompt: str) -> str: ...


class AnthropicClient:
    """Thin wrapper over the Anthropic SDK - the extractor never touches the
    SDK directly, so tests can substitute a fake without a network call."""

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        import anthropic  # imported lazily so importing this module never

        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, *, system: str, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )


GROQ_API = "https://api.groq.com/openai/v1/chat/completions"
GROQ_REQUEST_TIMEOUT = 30

# The three extraction passes hit Groq concurrently (see extract_candidate_
# claims below), which is exactly the shape that trips a free-tier tokens-
# per-minute limit - and a GitHub run with many repos can exceed the whole
# TPM budget for the run, not just for one call, since the same repo text is
# sent three times over (once per pass). A 429 there needs to survive across
# a full rate-limit window reset (Groq's Retry-After has been observed up to
# ~10s per wait), not just a couple of quick retries, so the ceiling here is
# generous rather than a real failure being blown up into a crash.
GROQ_MAX_ATTEMPTS = 8
GROQ_BACKOFF_BASE_SECONDS = 2.0
GROQ_BACKOFF_MAX_SECONDS = 60.0


def _groq_retry_delay(response, attempt: int) -> float:
    """Prefer the server's own Retry-After header (Groq sends one on 429);
    fall back to exponential backoff if it's absent or unparseable."""
    retry_after = response.headers.get("Retry-After") if response.headers else None
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return min(GROQ_BACKOFF_BASE_SECONDS * (2**attempt), GROQ_BACKOFF_MAX_SECONDS)


def _rate_limit_headers(response) -> dict[str, str]:
    """Every header Groq sent about the rate limit itself - not just
    Retry-After but the OpenAI-compatible x-ratelimit-* headers (remaining
    quota, reset time per limit type). Matched by prefix rather than an exact
    hardcoded name list, since which of these Groq actually sends has changed
    before and a prefix match stays correct either way. Logged (not just used
    for the retry math above) so a run that ultimately fails tells you WHEN
    it'll work again instead of just that it didn't."""
    if not response.headers:
        return {}
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() == "retry-after" or key.lower().startswith("x-ratelimit-")
    }


class GroqClient:
    """Thin wrapper over Groq's OpenAI-compatible chat-completions endpoint.
    Same `complete(system, prompt) -> str` contract as AnthropicClient, so
    swapping providers never touches _run_pass or extract_candidate_claims."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set - required to construct GroqClient."
            )
        self._api_key = key
        self._model = model
        self._http = session or requests
        self._sleep = sleep

    def complete(self, *, system: str, prompt: str) -> str:
        was_rate_limited = False
        for attempt in range(GROQ_MAX_ATTEMPTS):
            response = self._http.post(
                GROQ_API,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=GROQ_REQUEST_TIMEOUT,
            )
            is_last_attempt = attempt == GROQ_MAX_ATTEMPTS - 1
            if response.status_code == 429:
                headers = _rate_limit_headers(response)
                if not is_last_attempt:
                    delay = _groq_retry_delay(response, attempt)
                    logger.warning(
                        "[%s] Groq rate-limited, retrying in %.1fs (attempt %d/%d) - "
                        "rate-limit headers: %s",
                        _source_label.get(), delay, attempt + 2, GROQ_MAX_ATTEMPTS,
                        headers or "(none returned)",
                    )
                    status_callback = _status_callback.get()
                    if status_callback is not None:
                        # A calm, number-free sentence - api.py's status
                        # endpoint recomputes the actual remaining seconds
                        # fresh from rate_limit_resume_at on every poll, so
                        # the progress page never has to parse a number back
                        # out of this text (see RunStatus.rate_limit_resume_at).
                        status_callback(
                            detail="Rate-limited by the AI provider - waiting to retry",
                            rate_limit_resume_at=time.time() + delay,
                        )
                    was_rate_limited = True
                    self._sleep(delay)
                    continue
                logger.error(
                    "[%s] Groq rate limit exceeded after %d attempts - rate-limit "
                    "headers from the last response (retry-after / x-ratelimit-reset-* "
                    "say when it resets): %s",
                    _source_label.get(), GROQ_MAX_ATTEMPTS, headers or "(none returned)",
                )
            if was_rate_limited:
                # Clears the countdown regardless of whether this attempt is
                # the eventual success or the final, still-429 attempt that's
                # about to raise below - either way, this call is no longer
                # "waiting".
                status_callback = _status_callback.get()
                if status_callback is not None:
                    status_callback(rate_limit_resume_at=None)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        raise AssertionError("unreachable - loop always returns or raises")


def build_default_client() -> LLMClient:
    """
    The one place a provider is chosen. Explicit via LLM_PROVIDER=anthropic|
    groq; with no explicit setting, defaults to whichever API key is actually
    present (Anthropic first, since it's the tier this project started on).
    Never silently guesses on an unrecognized LLM_PROVIDER value - that's a
    config mistake, not a routing decision, so it fails loudly instead of
    picking one for you.
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "groq":
        return GroqClient()
    if provider:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}' - expected 'anthropic' or 'groq'."
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    if os.environ.get("GROQ_API_KEY"):
        return GroqClient()
    return AnthropicClient()


def validate_llm_config() -> None:
    """
    Fail loudly at startup, not on the first real request. Neither
    AnthropicClient nor the Anthropic SDK it wraps validates a missing/empty
    key at construction time (confirmed empirically: constructing one with
    no key at all succeeds - the SDK only rejects it on the first actual
    API call), so without this, a deploy with a blank ANTHROPIC_API_KEY looks
    fully healthy - the container starts, /health passes - right up until
    someone submits a real CV/GitHub/Upwork analysis and it fails with a
    generic SDK error instead of a clear one. Mirrors how
    app/storage/store.py's STORAGE_BACKEND=b2 already fails at import time if
    its required env vars are missing.

    Deliberately checks only PRESENCE, never makes a live call to confirm the
    key actually works - that's what a `live_api`-marked test is for, not a
    startup check that must stay fast and free. Mirrors build_default_client's
    exact provider-selection logic above (never adds a stricter rule of its
    own) so the two can't silently drift apart - same env vars, same
    precedence, just the fail case runs at startup instead of at first use.
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider == "anthropic":
        _require_llm_env("ANTHROPIC_API_KEY", "LLM_PROVIDER=anthropic")
        return
    if provider == "groq":
        _require_llm_env("GROQ_API_KEY", "LLM_PROVIDER=groq")
        return
    if provider:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{provider}' - expected 'anthropic' or 'groq'."
        )

    # No LLM_PROVIDER set: build_default_client() falls back to whichever key
    # is present (Anthropic first) - so at least one of the two must be set,
    # matching that fallback exactly rather than a stricter rule of our own.
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return
    if os.environ.get("GROQ_API_KEY", "").strip():
        return
    raise RuntimeError(
        "No LLM credentials configured - set LLM_PROVIDER=anthropic or "
        "LLM_PROVIDER=groq plus the matching API key (ANTHROPIC_API_KEY or "
        "GROQ_API_KEY), or set one of those two keys directly."
    )


def _require_llm_env(name: str, because: str) -> None:
    if not os.environ.get(name, "").strip():
        raise RuntimeError(f"{name} is not set (or is empty) - required because {because}.")


class _ExtractedSpan(BaseModel):
    """One candidate claim as the model reports it. Every field nullable -
    the model may find nothing for a given pass.

    Deliberately no start_index/end_index: the model is never asked to count
    characters (see module docstring for why that's unreliable). It quotes the
    supporting text verbatim instead, and _try_ground locates that quote in
    the source via exact string search - the only index math anywhere in this
    path is done by our own code, against the real string, not guessed by the
    model."""

    claim_text: str | None = None
    skill_ids: list[str] = []
    evidence_quote: str | None = None


class _ExtractionBatch(BaseModel):
    claims: list[_ExtractedSpan] = []


_PASS_INSTRUCTIONS: dict[str, str] = {
    "identity": "Find claims about role, title, or seniority (e.g. 'Senior backend engineer').",
    "history": "Find claims about specific projects or work history with concrete, checkable outcomes.",
    "skills": "Find claims that name a specific skill, tool, or technology the person used.",
}

# Which of the three passes above are actually worth running per source_type -
# each pass costs one LLM call, and running all three unconditionally is what
# turns a 20-repo GitHub profile into 60 calls, the dominant cost against a
# free-tier rate limit (CLAUDE.md, "Cut Groq calls substantially").
#
# GITHUB_REPO drops "identity": a repo README documents a PROJECT, not a
# person - it has no role/title/seniority language to find at all (the
# identity pass's own instruction above asks for "e.g. 'Senior backend
# engineer'", which does not occur in project documentation). This is not
# just a plausibility argument: _provisional_tier's own lookup table
# (_PROVISIONAL_TIER_BY_SOURCE, below) already assigns GITHUB_REPO claims T2
# regardless of which pass produced them - ingestion never even branches on
# pass_name for this source_type - so no downstream logic depends on an
# identity-pass result existing for a repo. UPWORK_TEXT drops "history": the
# pasted text is the freelancer's own overview/summary, which the identity
# and skills passes already cover (role framing, named tools); dedicated
# project-history detail lives in the CV and in GitHub's own repos, not
# reproduced a third time here. CV keeps all three - it is the one source
# that genuinely carries role, project history, AND a skills list, each
# worth its own targeted pass (and _provisional_tier's CV branch already
# depends on distinguishing the "history" pass from the "skills" pass).
#
# Any source_type not listed here (LINKEDIN_EXPORT, PORTFOLIO_SITE, etc. -
# none of which route_input dispatches to yet) falls back to all three passes
# via _DEFAULT_PASSES, matching the pre-existing behavior exactly rather than
# guessing at a narrower set for sources this codebase doesn't parse yet.
_DEFAULT_PASSES: tuple[str, ...] = ("identity", "history", "skills")
_PASSES_BY_SOURCE: dict[SourceType, tuple[str, ...]] = {
    SourceType.CV: ("identity", "history", "skills"),
    SourceType.UPWORK_TEXT: ("identity", "skills"),
    SourceType.GITHUB_REPO: ("history", "skills"),
}

_PROVISIONAL_TIER_BY_SOURCE: dict[SourceType, EvidenceTier] = {
    SourceType.GITHUB_REPO: EvidenceTier.T2,  # a repo is a demonstrated project
    SourceType.CV: EvidenceTier.T6,  # employer-confirmed work history, by default
    SourceType.UPWORK_TEXT: EvidenceTier.T8,  # self-declared, by default
}

# A pasted Upwork profile sometimes quotes an actual client review rather than
# just the freelancer's own summary. That's the one case where a source_type
# alone under-tiers the evidence - a quoted review naming a concrete outcome
# reads as client-verified (T1), not self-declared. Matched against the
# GROUNDED span text only (never the model's paraphrase), so this is still a
# rules-based check over evidence we already verified exists in the source.
_REVIEW_OUTCOME_PATTERN = re.compile(
    r"\b(review|testimonial|rated|star rating|\d\s*-\s*star|client (said|wrote|noted))\b",
    re.IGNORECASE,
)

# Generic package-manager install/run commands - identical in every project
# scaffolded from the same template, so naming one is never evidence the
# person built or demonstrated anything.
_BOILERPLATE_COMMAND_PATTERN = re.compile(
    r"\b(?:npm|yarn|pnpm|bun|npx)\s+(?:run\s+)?(?:dev|install|start|build|reset-project)\b",
    re.IGNORECASE,
)

# The model sometimes reports one of these as a bare "skill" claim, lifted
# straight out of a scaffolding command list or a framework's stock "run on
# a device" instructions - never a real accomplishment on its own.
_BARE_BOILERPLATE_TERMS = frozenset({
    "npm", "yarn", "pnpm", "bun", "npx",
    "expo go", "android emulator", "ios simulator", "development build",
})

# Literal phrases lifted from the actual default READMEs that `create-next-
# app` and `create-expo-app` generate. These are fingerprints of the
# boilerplate itself, not paraphrases of it, so a substring match is a safe,
# low-false-positive signal - unlike the bare terms above, ordinary project
# prose is very unlikely to reproduce this exact wording by coincidence.
_BOILERPLATE_PHRASES = (
    "bootstrapped with",
    "create-next-app",
    "create-expo-app",
    "welcome to your expo app",
    "auto-updates as you edit",
    "start editing the page by modifying",
    "you'll find options to open the app in a",
    "developing your project with expo",
    "easiest way to deploy your next.js app",
    "community of developers creating universal apps",
    "uses file-based routing",
    "move the starter code to the",
    "next.js deployment documentation",
    "take a look at the following resources",
    "your feedback and contributions are welcome",
)


def _is_boilerplate(text: str | None) -> bool:
    """
    Rules-based detector for generic framework-scaffolding text - e.g. the
    default README `create-next-app`/`create-expo-app` generate, describing
    install options and "getting started" steps every project of that type
    has. Deliberately NOT another LLM call: whether something is boilerplate
    must be as explainable and auditable as tier assignment, and a second
    model call would just relocate the "can the model be trusted here"
    problem instead of solving it.

    Applied to both the model's claim_text and the grounded span_text (see
    call site), since either one alone can carry the tell - a bare tool name
    as the claim, or a scaffolding phrase in the quote backing it.
    """
    if not text:
        return False
    lowered = text.lower().strip()
    if lowered in _BARE_BOILERPLATE_TERMS:
        return True
    if _BOILERPLATE_COMMAND_PATTERN.search(text):
        return True
    return any(phrase in lowered for phrase in _BOILERPLATE_PHRASES)


# --- Pre-send text preparation: strip boilerplate, then cap length ----------
#
# A separate, EARLIER step from _is_boilerplate above: _is_boilerplate filters
# a MODEL'S OUTPUT (a claim/quote) after an LLM call already happened; the
# functions below filter the INPUT before any call is made, so fewer tokens
# are spent on text that was never going to produce a real claim anyway.
# Seeded from the same _BOILERPLATE_PHRASES/_BOILERPLATE_COMMAND_PATTERN this
# module already uses for that later check - reused, not forked into a
# second, separately-maintained list.
#
# Deliberately rules-based, not model-based, for the same reason
# _is_boilerplate is: explainable and auditable, not a second "can the model
# be trusted here" problem.

# A markdown badge image, optionally wrapped in a link - `![alt](url)` or
# `[![alt](url)](url)`. A line consisting of nothing but one or more of these
# (badge rows are commonly several in a row, space-separated) is a visual
# status strip, never claim-worthy prose.
_BADGE_TOKEN_PATTERN = r"(?:\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!\[[^\]]*\]\([^)]*\))"
_BADGE_LINE_PATTERN = re.compile(rf"^\s*(?:{_BADGE_TOKEN_PATTERN}\s*)+$")

# A conservative, known-safe list of section headings whose content is
# reliably non-substantive boilerplate (license text, PR etiquette, a table
# of contents, a badge row under its own heading) - deliberately NOT a
# generic "drop anything under a heading that looks unimportant" heuristic,
# since guessing wrong there risks eating real project content.
_BOILERPLATE_SECTION_HEADING_PATTERN = re.compile(
    r"^#{1,6}\s*(license|licence|contributing|table of contents|badges?)\s*$",
    re.IGNORECASE,
)
_ANY_HEADING_PATTERN = re.compile(r"^#{1,6}\s+\S")

# A fenced code block ```...``` where every substantive line (blank lines and
# bare `#`-comment lines don't count against it) matches
# _BOILERPLATE_COMMAND_PATTERN is a generic install/run instructions block,
# identical across every project scaffolded from the same template - safe to
# drop entirely. A fence that mixes real content with commands is left alone.
_FENCE_PATTERN = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _strip_boilerplate_command_fences(text: str) -> str:
    def _replace(match: re.Match) -> str:
        body = match.group(1)
        significant_lines = [
            line for line in body.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if significant_lines and all(
            _BOILERPLATE_COMMAND_PATTERN.search(line) for line in significant_lines
        ):
            return ""
        return match.group(0)

    return _FENCE_PATTERN.sub(_replace, text)


def _strip_boilerplate_sections(text: str) -> str:
    """
    Remove known-boilerplate markup (badge lines, generic install/run code
    fences) and sections (license/contributing/TOC/badges headings and
    everything under them, up to the next heading) from `text`. Conservative
    by construction: only removes what matches an explicit, curated pattern -
    anything not recognized as boilerplate passes through untouched.
    """
    lines = text.splitlines()
    kept: list[str] = []
    skipping_section = False
    for line in lines:
        if skipping_section:
            if _ANY_HEADING_PATTERN.match(line):
                skipping_section = False
            else:
                continue
        if _BOILERPLATE_SECTION_HEADING_PATTERN.match(line.strip()):
            skipping_section = True
            continue
        if _BADGE_LINE_PATTERN.match(line):
            continue
        if any(phrase in line.lower() for phrase in _BOILERPLATE_PHRASES):
            continue
        kept.append(line)

    stripped = _strip_boilerplate_command_fences("\n".join(kept))
    # Collapse runs of blank lines left behind by removed sections/fences -
    # cosmetic, and saves a few more characters.
    return re.sub(r"\n{3,}", "\n\n", stripped).strip()


# How many characters of (already boilerplate-stripped) document text are
# actually sent to the LLM per call - a safety net against outlier-large
# documents costing tokens disproportionate to what they add. 6000 chars
# (~1,500 tokens) comfortably covers the vast majority of real CVs/READMEs -
# this module's own docstring cites 7,898 characters as the largest README
# measured against this codebase - while bounding the worst case.
# MAX_EXTRACTION_CHARS overrides this default, never a code change; tune it
# from real measured call/char counts (see the token-reduction plan's "A0"
# step) rather than guessing further.
_MAX_EXTRACTION_CHARS_ENV = "MAX_EXTRACTION_CHARS"
DEFAULT_MAX_EXTRACTION_CHARS = 6000


def _extraction_char_limit() -> int:
    """MAX_EXTRACTION_CHARS overrides DEFAULT_MAX_EXTRACTION_CHARS; unset/
    blank keeps the default. Fails loudly on a non-positive-integer value,
    matching GITHUB_REPO_CAP's own convention (see github_parser.py)."""
    raw = os.environ.get(_MAX_EXTRACTION_CHARS_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_EXTRACTION_CHARS
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_MAX_EXTRACTION_CHARS_ENV} must be an integer, got {raw!r}"
        ) from exc
    if limit <= 0:
        raise ValueError(f"{_MAX_EXTRACTION_CHARS_ENV} must be a positive integer, got {limit}")
    return limit


def _cap_text(text: str, limit: int) -> str:
    """Truncate `text` to at most `limit` characters, preferring to cut at a
    paragraph or word boundary near the limit rather than mid-word - purely
    cosmetic (a truncated word costs nothing extra downstream), but avoids an
    ugly half-word cutoff for no reason. Never appends any marker text: the
    returned string is exactly what gets sent to the LLM AND stored as the
    document a claim's span grounds against (see prepare_extraction_text), so
    it must contain nothing that wasn't literally in the source."""
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    break_point = truncated.rfind("\n\n")
    if break_point == -1:
        break_point = truncated.rfind(" ")
    if break_point > limit * 0.5:  # don't discard most of the text chasing a boundary
        truncated = truncated[:break_point]
    return truncated.rstrip()


def prepare_extraction_text(text: str) -> str:
    """
    Apply the boilerplate stripper, then the hard character cap, to `text`
    BEFORE it is stored (file_store.put_source) or sent to the LLM as a
    prompt (extract_candidate_claims's `text` argument).

    Both call sites (app/ingestion/github_parser.py, app/ingestion/
    cv_parser.py) MUST feed this same, already-prepared string into both
    put_source and extract_candidate_claims - never the original raw text to
    one and the prepared text to the other. A claim's span indices are
    computed against whatever string was actually sent to the LLM; if a
    document is stored under a DIFFERENT string, those indices point at the
    wrong text the next time it's read back (e.g. reverify_span in
    app/generation/validator.py re-reading the stored document to confirm a
    generated asset's numbers are still backed by real evidence).
    """
    stripped = _strip_boilerplate_sections(text)
    return _cap_text(stripped, _extraction_char_limit())


def _provisional_tier(source_type: SourceType, pass_name: str, span_text: str) -> EvidenceTier:
    """
    THE ONLY PLACE a Claim's evidence_tier is decided at extraction time.

    Rules-based and fully explainable: a lookup on source_type, refined by
    which extraction pass produced the claim and, for Upwork, a keyword check
    against the grounded text. The model that ran the pass never supplies a
    tier - `_ExtractedSpan` has no such field - so there is nothing here for
    it to leak. This is deliberately provisional; `assign_tiers` (stage 3)
    later refines these using cross-source corroboration.
    """
    if source_type == SourceType.CV and pass_name == "skills":
        return EvidenceTier.T8  # a bare skill mention, not confirmed history
    if source_type == SourceType.UPWORK_TEXT and _REVIEW_OUTCOME_PATTERN.search(span_text):
        return EvidenceTier.T1  # a quoted client review naming an outcome
    return _PROVISIONAL_TIER_BY_SOURCE.get(source_type, EvidenceTier.T8)


_RESPONSE_FORMAT_INSTRUCTIONS = (
    'Return ONLY JSON matching {"claims": [{"claim_text": str, '
    '"skill_ids": [str], "evidence_quote": str}]}. '
    "evidence_quote MUST be copied character-for-character from the EXACT "
    "text given by the user - the literal substring that supports the "
    "claim. Do not paraphrase, summarize, fix typos, add ellipses, or "
    "change whitespace or punctuation in any way - it must be an exact, "
    "verbatim, contiguous quote that could be found with a plain text "
    "search. Keep it as short as possible while still supporting the "
    'claim. If nothing qualifies, return {"claims": []}. No prose, no '
    "markdown fences - JSON only."
)


def _system_prompt(pass_name: str) -> str:
    instruction = _PASS_INSTRUCTIONS[pass_name]
    return (
        f"You extract claims from freelancer profile text. {instruction}\n"
        f"{_RESPONSE_FORMAT_INSTRUCTIONS}"
    )


# Source types where every pass listed for them in _PASSES_BY_SOURCE can be
# folded into ONE LLM call instead of one call per pass - cutting both the
# request count AND (since each call was re-sending the same document text)
# the token volume of a run. Safe specifically because _provisional_tier
# never branches on WHICH pass produced a claim for either of these two
# source types (GITHUB_REPO always resolves via _PROVISIONAL_TIER_BY_SOURCE
# alone; UPWORK_TEXT's one pass-independent refinement is a text-pattern
# match, not a pass_name check) - see _provisional_tier's docstring. Nothing
# downstream needs to know which instruction produced a given claim for
# either of them.
#
# CV is deliberately NOT included: _provisional_tier's one real pass_name
# branch (source_type == CV and pass_name == "skills" -> T8) is
# tier-determinative, and merging would require the model to self-report
# which category a claim belongs to - a small but real erosion of the
# invariant that "the model that ran the pass never supplies a tier"
# (_ExtractedSpan has no such field). CV is also the cheapest source per run
# (3 calls total, never multiplied by a repo count), so the risk isn't worth
# the reward here - see CLAUDE.md-adjacent plan notes on this exact tradeoff.
_MERGEABLE_PASS_SOURCES = frozenset({SourceType.GITHUB_REPO, SourceType.UPWORK_TEXT})


def _merged_system_prompt(pass_names: tuple[str, ...]) -> str:
    """One system prompt covering every pass in `pass_names` at once, for
    source types in _MERGEABLE_PASS_SOURCES. Each pass's own instruction text
    is included verbatim (not paraphrased or summarized) so a single merged
    call still asks for exactly what the separate calls would have asked for,
    just once instead of once per pass."""
    instructions = " ".join(_PASS_INSTRUCTIONS[name] for name in pass_names)
    return (
        f"You extract claims from freelancer profile text. {instructions}\n"
        f"{_RESPONSE_FORMAT_INSTRUCTIONS}"
    )


def _run_pass(client: LLMClient, label: str, system: str, text: str) -> _ExtractionBatch:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            raw = client.complete(system=system, prompt=text)
            data = json.loads(raw)
            return _ExtractionBatch.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            remaining_attempts = RETRY_ATTEMPTS - attempt - 1
            if remaining_attempts > 0:
                logger.warning(
                    "[%s] pass '%s': malformed response, retrying (attempt %d/%d)",
                    _source_label.get(), label, attempt + 2, RETRY_ATTEMPTS,
                )
            continue
    # Every attempt came back malformed - contribute nothing rather than risk
    # a claim built on data that didn't pass its own schema.
    logger.warning(
        "[%s] pass '%s': gave up after %d malformed responses",
        _source_label.get(), label, RETRY_ATTEMPTS,
    )
    return _ExtractionBatch(claims=[])


def extract_candidate_claims(
    client: LLMClient,
    *,
    document_id: str,
    text: str,
    source_type: SourceType,
    locator_prefix: str | None = None,
    observed_date: date | None = None,
) -> list[Claim]:
    """
    Run the identity/history/skills passes in parallel over `text`, ground
    every returned span against `text`, and return the resulting Claims.

    `observed_date` (e.g. a repo's last commit date) is passed straight
    through to every Claim from this call; recency_factor is derived from it
    via the documented decay function in config/weights, never guessed here.
    """
    ungrounded_weight = TIER_WEIGHTS[EvidenceTier.T8.value]
    factor = compute_recency_factor(observed_date)

    _source_label.set(locator_prefix or source_type.value)

    pass_names = _PASSES_BY_SOURCE.get(source_type, _DEFAULT_PASSES)

    if source_type in _MERGEABLE_PASS_SOURCES and len(pass_names) > 1:
        # One call covering every pass at once instead of one call per pass -
        # see _MERGEABLE_PASS_SOURCES for why this is safe for exactly these
        # source types. `label` is only ever used for log messages and as the
        # `pass_name` argument to _provisional_tier below, which (for both
        # source types in this branch) never actually branches on its value.
        label = "+".join(pass_names)
        batches = [(label, _run_pass(client, label, _merged_system_prompt(pass_names), text))]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pass_names)) as pool:
            # A fresh copy_context() per submission, not one shared Context
            # reused across all of them - a single Context object raises
            # RuntimeError ("already entered") if .run() executes
            # concurrently from more than one thread at once, which
            # simultaneous passes would trigger immediately. Independent
            # copies of the same snapshotted value are safe to run
            # concurrently; a single shared one is not (verified).
            futures = {
                pool.submit(
                    contextvars.copy_context().run,
                    _run_pass, client, pass_name, _system_prompt(pass_name), text,
                ): pass_name
                for pass_name in pass_names
            }
            batches = [
                (futures[future], future.result())
                for future in concurrent.futures.as_completed(futures)
            ]

    claims: list[Claim] = []
    for pass_name, batch in batches:
        for item in batch.claims:
            if not item.claim_text:
                continue
            if _is_boilerplate(item.claim_text):
                continue  # e.g. claim_text is just "npm" - excluded, not even a coaching prompt
            span = _try_ground(
                item, document_id=document_id, text=text, locator=locator_prefix
            )
            if span is not None and _is_boilerplate(span.text):
                continue  # the quote itself is scaffolding prose, not evidence
            tier = (
                _provisional_tier(source_type, pass_name, span.text)
                if span is not None
                else EvidenceTier.T8  # ungrounded: weakest tier, no exception
            )
            claims.append(
                Claim(
                    claim_text=item.claim_text,
                    skill_ids=item.skill_ids,
                    source_type=source_type,
                    source_span=span,
                    evidence_tier=tier,
                    weight=TIER_WEIGHTS[tier.value] if span is not None else ungrounded_weight,
                    observed_date=observed_date,
                    recency_factor=factor,
                )
            )
    return _dedupe_claims(claims)


# Two claims collapse into one above this string-similarity ratio - high
# enough that genuinely different accomplishments (different quotes above in
# the boilerplate tests, different sentences in general) stay separate, low
# enough to catch the same accomplishment reworded slightly between passes.
# Plain difflib ratio on normalized text, deliberately not an LLM call or an
# embedding - "good enough to stop obvious duplication" per the brief, not a
# general-purpose dedup guarantee.
NEAR_DUPLICATE_SIMILARITY_THRESHOLD = 0.6


def _normalize_claim_text(text: str) -> str:
    return " ".join(text.split()).lower()


def _claim_text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _is_better_duplicate(candidate: Claim, current: Claim) -> bool:
    """
    Deterministic tie-break for which of two duplicate claims survives -
    never "whichever pass happened to finish first" (thread completion order
    from the executor above is not something callers should be able to
    observe). Grounded beats ungrounded; among grounded claims, higher
    effective_weight (tier and recency combined) wins; a final tie prefers
    the longer grounded span as the more complete piece of evidence.
    """
    candidate_grounded = candidate.source_span is not None
    current_grounded = current.source_span is not None
    if candidate_grounded != current_grounded:
        return candidate_grounded
    if not candidate_grounded:
        return False  # both ungrounded - keep whichever was found first
    if candidate.effective_weight != current.effective_weight:
        return candidate.effective_weight > current.effective_weight
    return len(candidate.source_span.text) > len(current.source_span.text)


def _dedupe_claims(claims: list[Claim]) -> list[Claim]:
    """
    Collapse claims the three parallel passes (identity/history/skills)
    independently rediscovered. Two failure modes this fixes:

    1. Exact duplicates: the same accomplishment, reported with identical
       claim_text by more than one pass (normalized for whitespace/case).
    2. Near-duplicates: the same accomplishment described in overlapping but
       not identical wording (e.g. the history pass and the skills pass each
       phrase the same repo highlight slightly differently) - caught via a
       string-similarity threshold rather than an LLM call, per the brief.

    Comparison is against every claim already kept, not just the most recent
    one, so three-plus near-identical variants collapse to one regardless of
    which order the passes finished in.
    """
    kept: list[Claim] = []
    for claim in claims:
        normalized = _normalize_claim_text(claim.claim_text)
        merged = False
        for index, existing in enumerate(kept):
            existing_normalized = _normalize_claim_text(existing.claim_text)
            is_duplicate = (
                normalized == existing_normalized
                or _claim_text_similarity(normalized, existing_normalized)
                >= NEAR_DUPLICATE_SIMILARITY_THRESHOLD
            )
            if is_duplicate:
                if _is_better_duplicate(claim, existing):
                    kept[index] = claim
                merged = True
                break
        if not merged:
            kept.append(claim)
    return kept


# Below this many characters, a quote is too generic to trust as a location -
# a short common word/phrase risks matching the wrong occurrence in the
# source even though it's technically a verbatim substring somewhere.
MIN_QUOTE_CHARS = 8

# A model reliably reproduces a quoted phrase's WORDS correctly but not
# always its exact punctuation glyph - e.g. it wraps the phrase in straight
# single quotes where the source uses curly DOUBLE quotes (a real observed
# case - the model isn't just picking a different style of the SAME quote
# family, it mixes families too), or uses a plain hyphen where the source
# has an en/em dash. That's real, provable evidence lost to nothing more
# than character-set noise, not paraphrasing (measured on a real run: ~40%
# of ungrounded claims - 12 of 30 - were this exact failure mode). Every
# quote variant - single or double, curly or straight - folds to the SAME
# canonical character (a plain apostrophe), not two separate canonical forms
# segregated by family, specifically so a single-vs-double mismatch like the
# one above still matches. Dashes get their own, separate canonical form.
#
# Every key/value pair here is a single Unicode code point mapped to a
# single plain-ASCII code point - never a multi-character substitution - so
# normalizing never changes string length or shifts character positions.
# That's what makes it safe to use for locating an index into the ORIGINAL,
# un-normalized source text below (see _try_ground): this is character-class
# canonicalization over a small fixed set of known typographically-
# equivalent glyphs, not fuzzy/approximate matching - every other character
# still must match exactly.
_QUOTE_AND_DASH_NORMALIZATION = str.maketrans({
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "’": "'",  # RIGHT SINGLE QUOTATION MARK
    "‚": "'",  # SINGLE LOW-9 QUOTATION MARK
    "‛": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    '"': "'",  # QUOTATION MARK (straight double)
    "“": "'",  # LEFT DOUBLE QUOTATION MARK
    "”": "'",  # RIGHT DOUBLE QUOTATION MARK
    "„": "'",  # DOUBLE LOW-9 QUOTATION MARK
    "‟": "'",  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "–": "-",  # EN DASH
    "—": "-",  # EM DASH
})


def _normalize_quotes_and_dashes(value: str) -> str:
    """Fold typographically-equivalent quote/dash glyphs to one plain-ASCII
    form each. Used only to LOCATE a match (see _try_ground) - the span's
    actual text always comes from re-slicing the original, un-normalized
    source, never this normalized copy."""
    return value.translate(_QUOTE_AND_DASH_NORMALIZATION)

_STOPWORDS = frozenset({
    "with", "that", "this", "from", "were", "have", "been", "into", "over",
    "using", "used", "your", "their", "which", "about", "these", "those",
    "will", "would", "could", "should", "there", "where", "while",
})


def _significant_words(text: str) -> set[str]:
    """Lowercase words of at least 4 characters, minus common stopwords -
    a deliberately crude vocabulary fingerprint, only ever used to check that
    two strings are plausibly *about* the same thing, never to ground a span
    itself (that's still exact string search)."""
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{3,}", text)
        if word.lower() not in _STOPWORDS
    }


def _try_ground(
    item: _ExtractedSpan, *, document_id: str, text: str, locator: str | None
) -> SourceSpan | None:
    """
    Locate item.evidence_quote in `text` via exact string search and ground
    it - this is the ONLY place a span's indices are computed. The model
    never supplies start_index/end_index; it only supplies a quote, and
    `str.find` (not the model) does the arithmetic. If the quote isn't an
    exact substring - the model paraphrased, even slightly - str.find returns
    -1 and the claim drops to ungrounded rather than being published against
    a guessed location. When a quote appears more than once, the first
    occurrence is used - deterministic, and good enough for this slice.

    A verbatim quote existing somewhere in the source is necessary but not
    sufficient: nothing stops a model from inventing a claim (e.g. an
    "identity" pass finding no actual role/title in a GitHub README) and
    attaching some unrelated-but-real quote from elsewhere in the document
    just to satisfy the schema. The vocabulary-overlap check below catches
    that: if the claim and its supposed evidence share not even one
    significant word, the "evidence" almost certainly isn't evidence for
    THIS claim, so it's dropped rather than published as if it were proof.

    The search itself runs against quote-and-dash-normalized copies of both
    strings (see _QUOTE_AND_DASH_NORMALIZATION) so a model that reproduces a
    quote's words correctly but with a different quote glyph (curly vs
    straight) or dash (en/em vs hyphen) still grounds - that's typographic
    noise, not a paraphrase. Still exact substring matching, just over a
    canonicalized character set; every other character must match exactly,
    and the normalization is strictly one-code-point-to-one-code-point, so
    the index found in the normalized text is valid in the ORIGINAL `text`
    too - which is what actually gets sliced, below and in ground_span.
    """
    quote = item.evidence_quote
    if not quote or len(quote) < MIN_QUOTE_CHARS:
        return None
    normalized_quote = _normalize_quotes_and_dashes(quote)
    start_index = _normalize_quotes_and_dashes(text).find(normalized_quote)
    if start_index == -1:
        return None
    end_index = start_index + len(normalized_quote)
    try:
        span = ground_span(
            document_id=document_id,
            source_text=text,
            start_index=start_index,
            end_index=end_index,
            locator=locator,
        )
    except SpanGroundingError:
        return None
    if not (_significant_words(item.claim_text) & _significant_words(span.text)):
        return None
    return span
