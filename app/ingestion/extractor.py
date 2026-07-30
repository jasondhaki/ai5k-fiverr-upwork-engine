"""
The parallel LLM extractor: turns raw source text into candidate claims.

Three small, independent passes run per document - identity, history, skills -
each schema-validated and retried once on failure. The model never returns
claim text to trust blindly: it returns a verbatim quote copied from the exact
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
            if response.status_code == 429 and not is_last_attempt:
                delay = _groq_retry_delay(response, attempt)
                logger.warning(
                    "[%s] Groq rate-limited, retrying in %.1fs (attempt %d/%d)",
                    _source_label.get(), delay, attempt + 2, GROQ_MAX_ATTEMPTS,
                )
                self._sleep(delay)
                continue
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


def _system_prompt(pass_name: str) -> str:
    instruction = _PASS_INSTRUCTIONS[pass_name]
    return (
        f"You extract claims from freelancer profile text. {instruction}\n"
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


def _run_pass(client: LLMClient, pass_name: str, text: str) -> _ExtractionBatch:
    system = _system_prompt(pass_name)
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
                    _source_label.get(), pass_name, attempt + 2, RETRY_ATTEMPTS,
                )
            continue
    # Every attempt came back malformed - contribute nothing rather than risk
    # a claim built on data that didn't pass its own schema.
    logger.warning(
        "[%s] pass '%s': gave up after %d malformed responses",
        _source_label.get(), pass_name, RETRY_ATTEMPTS,
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(_PASS_INSTRUCTIONS)) as pool:
        # A fresh copy_context() per submission, not one shared Context reused
        # across all three - a single Context object raises RuntimeError
        # ("already entered") if .run() executes concurrently from more than
        # one thread at once, which three simultaneous passes would trigger
        # immediately. Independent copies of the same snapshotted value are
        # safe to run concurrently; a single shared one is not (verified).
        futures = {
            pool.submit(contextvars.copy_context().run, _run_pass, client, pass_name, text): pass_name
            for pass_name in _PASS_INSTRUCTIONS
        }
        batches = [(futures[future], future.result()) for future in concurrent.futures.as_completed(futures)]

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
    """
    quote = item.evidence_quote
    if not quote or len(quote) < MIN_QUOTE_CHARS:
        return None
    start_index = text.find(quote)
    if start_index == -1:
        return None
    end_index = start_index + len(quote)
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
