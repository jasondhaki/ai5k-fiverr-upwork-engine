"""
Tests for the parallel LLM extractor.

The behavior that matters most: a claim whose evidence_quote doesn't ground
against the source text NEVER gets a source_span, so it can never be
published as if it were proven. Everything else (retries, empty passes) is
secondary.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date

import pytest
import requests

import app.ingestion.extractor as extractor
from app.ingestion.extractor import (
    RETRY_ATTEMPTS,
    GroqClient,
    _ExtractedSpan,
    _is_boilerplate,
    _provisional_tier,
    _significant_words,
    _try_ground,
    build_default_client,
    extract_candidate_claims,
    status_reporting,
    validate_llm_config,
)
from app.schemas import Claim, SourceType
from app.schemas.claim import EvidenceTier

SOURCE = "Shipped a fraud-detection model to production, cutting false positives by 22%."


class _StaticClient:
    """Returns the same JSON body for every pass - identity, history, skills
    all see it, so a single claim in the body is independently rediscovered
    three times over before the dedup step collapses it back to one."""

    def __init__(self, body: str) -> None:
        self.body = body
        self.calls = 0

    def complete(self, *, system: str, prompt: str) -> str:
        self.calls += 1
        return self.body


class _PerPassClient:
    """Returns a different fixed body depending on which pass (identity/
    history/skills) is asking, keyed by matching that pass's instruction
    text against the system prompt. Lets a test give each pass its own
    distinct, verbatim-groundable claim - needed once dedup exists, since the
    old "same body for every pass" pattern is now exactly what collapses."""

    def __init__(self, bodies_by_pass: dict[str, str]) -> None:
        self._bodies_by_pass = bodies_by_pass

    def complete(self, *, system: str, prompt: str) -> str:
        for pass_name, instruction in extractor._PASS_INSTRUCTIONS.items():
            if instruction in system:
                return self._bodies_by_pass[pass_name]
        raise AssertionError("system prompt didn't match any known pass instruction")


class _FlakyOnceClient:
    """Fails JSON parsing on exactly the first call across all threads, then
    succeeds on every subsequent call - proves the retry actually runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self.calls = 0

    def complete(self, *, system: str, prompt: str) -> str:
        with self._lock:
            self._count += 1
            self.calls += 1
            first = self._count == 1
        return "not json" if first else json.dumps({"claims": []})


def test_valid_span_is_grounded_and_published():
    body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "cut false positives 22%",
                    "skill_ids": ["fraud_detection"],
                    "evidence_quote": "cutting false positives by 22%",
                }
            ]
        }
    )
    claims = extract_candidate_claims(
        _StaticClient(body),
        document_id="doc-1",
        text=SOURCE,
        source_type=SourceType.GITHUB_REPO,
    )

    # identity/history/skills all saw the same body and would each produce
    # this claim independently, but the dedup step collapses the exact
    # repeat down to one - the profile shows this accomplishment once.
    assert len(claims) == 1
    assert all(c.publishable for c in claims)
    assert all(c.source_span.text == "cutting false positives by 22%" for c in claims)


def test_a_quote_that_isnt_a_verbatim_substring_never_becomes_a_source_span():
    """
    The model is never asked for indices - only a quote - so the failure mode
    this guards against is a paraphrase (even a slight one) rather than an
    out-of-bounds index. str.find returns -1 for anything that isn't an exact
    substring, and that must drop the claim to ungrounded, not guess a nearby
    location.
    """
    body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "invented number",
                    "skill_ids": [],
                    "evidence_quote": "cutting false positives by twenty-two percent",
                }
            ]
        }
    )
    claims = extract_candidate_claims(
        _StaticClient(body),
        document_id="doc-1",
        text=SOURCE,
        source_type=SourceType.GITHUB_REPO,
    )

    # all three passes produced the identical ungrounded claim; dedup
    # collapses the exact repeat down to one regardless of grounding status
    assert len(claims) == 1
    assert all(c.source_span is None for c in claims)
    assert all(not c.publishable for c in claims)


def test_a_quote_too_short_to_trust_never_becomes_a_source_span():
    """A short, generic quote is technically a verbatim substring but too
    weak to trust as a location - it risks grounding against the wrong
    occurrence of common text."""
    body = json.dumps(
        {"claims": [{"claim_text": "vague", "skill_ids": [], "evidence_quote": "a"}]}
    )
    claims = extract_candidate_claims(
        _StaticClient(body),
        document_id="doc-1",
        text=SOURCE,
        source_type=SourceType.GITHUB_REPO,
    )
    assert all(c.source_span is None for c in claims)


def test_missing_evidence_quote_is_not_published():
    body = json.dumps({"claims": [{"claim_text": "vague claim", "skill_ids": []}]})
    claims = extract_candidate_claims(
        _StaticClient(body),
        document_id="doc-1",
        text=SOURCE,
        source_type=SourceType.UPWORK_TEXT,
    )

    assert claims
    assert all(c.source_span is None for c in claims)


def test_empty_pass_contributes_nothing():
    claims = extract_candidate_claims(
        _StaticClient(json.dumps({"claims": []})),
        document_id="doc-1",
        text=SOURCE,
        source_type=SourceType.CV,
    )
    assert claims == []


def test_malformed_json_exhausts_retries_and_contributes_nothing():
    client = _StaticClient("not valid json")
    claims = extract_candidate_claims(
        client, document_id="doc-1", text=SOURCE, source_type=SourceType.CV
    )
    assert claims == []
    # 3 passes, each retried RETRY_ATTEMPTS times before giving up
    assert client.calls == 3 * RETRY_ATTEMPTS


def test_retry_recovers_after_one_malformed_response():
    client = _FlakyOnceClient()
    claims = extract_candidate_claims(
        client, document_id="doc-1", text=SOURCE, source_type=SourceType.CV
    )
    assert claims == []  # the recovered response is an empty claims list
    # exactly one pass needed its retry; the other two succeeded first try
    assert client.calls == 4


# --- Quote/dash normalization: typographic noise isn't a paraphrase ---------
#
# ~40% of ungrounded claims in a real run (12 of 30) were this exact failure
# mode: the model reproduces a quoted phrase's words correctly but with a
# different quote glyph than the source uses, and str.find missed an
# otherwise perfect match. Real, provable evidence, discarded for nothing
# more than punctuation noise.


def test_a_quote_wrapped_in_different_quote_glyphs_than_the_source_still_grounds():
    """The exact reported case: the source wraps the phrase in curly DOUBLE
    quotes, the model's evidence_quote wraps it in straight SINGLE quotes -
    not just a different style of the same quote family, a different family
    entirely. Must still ground, and the grounded span.text must be the
    ORIGINAL source's exact characters (curly quotes and all) - never the
    normalized copy used only to locate it."""
    source = (
        "The site headline: “BRIDGING THE GAP BETWEEN PIXELS AND "
        "PNEUMATICS” ran across the top of the page."
    )
    item = _ExtractedSpan(
        claim_text="Wrote the site's PIXELS AND PNEUMATICS headline",
        evidence_quote="'BRIDGING THE GAP BETWEEN PIXELS AND PNEUMATICS'",
    )

    span = _try_ground(item, document_id="doc-1", text=source, locator=None)

    assert span is not None
    assert span.text == (
        "“BRIDGING THE GAP BETWEEN PIXELS AND PNEUMATICS”"
    )
    # The span's indices must resolve to that exact substring in the
    # ORIGINAL, un-normalized source - not a coincidence of the normalized copy.
    assert source[span.start_index : span.end_index] == span.text


def test_an_en_dash_in_the_quote_still_grounds_against_a_hyphen_in_the_source():
    source = "Rebuilt the pipeline end-to-end, cutting latency significantly."
    item = _ExtractedSpan(
        claim_text="Rebuilt the pipeline",
        evidence_quote="Rebuilt the pipeline end–to–end",  # en dashes
    )

    span = _try_ground(item, document_id="doc-1", text=source, locator=None)

    assert span is not None
    assert span.text == "Rebuilt the pipeline end-to-end"  # source's real hyphens


def test_a_genuinely_different_phrase_still_fails_to_ground():
    """Normalization must not become a backdoor for approximate matching -
    only the fixed, known set of quote/dash glyphs is folded together, so a
    real paraphrase (different words, not just different punctuation) still
    fails exactly as before."""
    source = (
        "The site headline: “BRIDGING THE GAP BETWEEN PIXELS AND "
        "PNEUMATICS” ran across the top of the page."
    )
    item = _ExtractedSpan(
        claim_text="Wrote a completely different headline",
        evidence_quote="CROSSING THE CHASM FROM SENSORS TO SOFTWARE",
    )

    assert _try_ground(item, document_id="doc-1", text=source, locator=None) is None


# --- Deduplication: one accomplishment shouldn't inflate a profile ----------
#
# Fixtures below use the actual repeated claim reported from a real run
# against the digital-workshop repo: "Implemented a responsive 3D wireframe
# environment using React Three Fiber and Drei", surfaced by more than one
# pass with different spans attached. identity/history/skills run
# independently over the SAME text, so the same real accomplishment is
# routinely rediscovered by more than one of them - that's expected, and the
# dedup step (not the model) is what collapses it back to one claim.

_DIGITAL_WORKSHOP_CLAIM = (
    "Implemented a responsive 3D wireframe environment using React Three "
    "Fiber and Drei"
)
_DIGITAL_WORKSHOP_README = (
    "Workshop Canvas: Scaffolded the main 3D wireframe environment using "
    "React Three Fiber and Drei, where 3D models subtly rotate and follow "
    "the user's cursor.\n\n"
    "System Terminal: Built a 3D wireframe environment with React Three "
    "Fiber and Drei that responds to the cursor, alongside a live-typing "
    "status window.\n\n"
    "Skills Orbit: Established a scalable data structure in data/skills.ts "
    "that defines technical skills and their motherboard interconnections.\n\n"
    "Created the SkillNode.tsx component using Lucide React icons and "
    "Framer Motion for smooth, physics-based scaling."
)


def test_exact_duplicate_claim_text_collapses_to_one():
    """The real reported bug: the same claim_text, verbatim, from two
    different passes with two DIFFERENT (both validly grounded) spans - the
    kind of duplication that would previously show up twice with different,
    and per the separate offset bug, possibly wrong, spans attached."""
    client = _PerPassClient(
        {
            "identity": json.dumps({"claims": []}),
            "history": _single_claim_body(
                _DIGITAL_WORKSHOP_CLAIM,
                "Scaffolded the main 3D wireframe environment using React Three "
                "Fiber and Drei",
            ),
            "skills": _single_claim_body(
                _DIGITAL_WORKSHOP_CLAIM,
                "3D wireframe environment with React Three Fiber and Drei that "
                "responds to the cursor",
            ),
        }
    )
    claims = extract_candidate_claims(
        client,
        document_id="doc-1",
        text=_DIGITAL_WORKSHOP_README,
        source_type=SourceType.GITHUB_REPO,
    )

    assert len(claims) == 1
    assert claims[0].claim_text == _DIGITAL_WORKSHOP_CLAIM
    assert claims[0].publishable


def test_near_duplicate_reworded_claim_collapses_to_one():
    """Same accomplishment, described in genuinely different wording by two
    passes - no shared claim_text to key off of, only similarity."""
    client = _PerPassClient(
        {
            "identity": json.dumps({"claims": []}),
            "history": _single_claim_body(
                _DIGITAL_WORKSHOP_CLAIM,
                "Scaffolded the main 3D wireframe environment using React Three "
                "Fiber and Drei",
            ),
            "skills": _single_claim_body(
                "Built a 3D wireframe environment with React Three Fiber and "
                "Drei that responds to the cursor",
                "3D wireframe environment with React Three Fiber and Drei that "
                "responds to the cursor",
            ),
        }
    )
    claims = extract_candidate_claims(
        client,
        document_id="doc-1",
        text=_DIGITAL_WORKSHOP_README,
        source_type=SourceType.GITHUB_REPO,
    )

    assert len(claims) == 1  # two different phrasings of one accomplishment
    assert claims[0].publishable


def test_distinct_real_accomplishments_are_never_merged():
    """Two genuinely different, unrelated accomplishments from the same real
    repo must both survive - dedup must not over-merge just because they
    came from the same document."""
    client = _PerPassClient(
        {
            "identity": json.dumps({"claims": []}),
            "history": _single_claim_body(
                "established a scalable skills data structure",
                "Established a scalable data structure in data/skills.ts",
            ),
            "skills": _single_claim_body(
                "built the SkillNode component with Framer Motion",
                "Created the SkillNode.tsx component using Lucide React icons "
                "and Framer Motion",
            ),
        }
    )
    claims = extract_candidate_claims(
        client,
        document_id="doc-1",
        text=_DIGITAL_WORKSHOP_README,
        source_type=SourceType.GITHUB_REPO,
    )

    assert len(claims) == 2
    assert all(c.publishable for c in claims)


def test_dedupe_claims_keeps_the_grounded_claim_over_the_ungrounded_duplicate():
    """Direct unit test of the tie-break rule: given a choice between two
    duplicates, the grounded one always wins, regardless of arrival order."""
    from app.ingestion.extractor import _dedupe_claims
    from app.ingestion.span_grounding import ground_span

    grounded_span = ground_span(
        document_id="doc-1", source_text=SOURCE, start_index=0, end_index=16
    )
    ungrounded = Claim(
        claim_text="shipped a fraud model",
        source_type=SourceType.GITHUB_REPO,
        source_span=None,
        evidence_tier=EvidenceTier.T8,
        weight=0.15,
    )
    grounded = Claim(
        claim_text="shipped a fraud model",
        source_type=SourceType.GITHUB_REPO,
        source_span=grounded_span,
        evidence_tier=EvidenceTier.T2,
        weight=0.85,
    )

    assert [c.source_span for c in _dedupe_claims([ungrounded, grounded])] == [grounded_span]
    assert [c.source_span for c in _dedupe_claims([grounded, ungrounded])] == [grounded_span]


# --- Boilerplate filtering: scaffolding text is not a claim -----------------
#
# Every project bootstrapped from the same framework template ships identical
# install instructions and "getting started" prose. The extractor was
# surfacing that as claims - "npm", "yarn", "Expo Go", "Android emulator" -
# lifted straight from create-next-app/create-expo-app's default README, as
# if naming a package manager were something the freelancer built. The
# vocabulary-overlap check above doesn't catch this: the model's claim_text
# ("uses yarn") and its quote ("yarn dev") share a word perfectly well, so
# overlap alone says nothing about whether the underlying text is generic
# scaffolding rather than a real accomplishment. This needs its own,
# separately rules-based filter - never a second LLM call, for the same
# explainability reason tier assignment isn't model-based.


def test_bare_tool_names_are_boilerplate():
    for term in ("npm", "Yarn", "PNPM", "bun", "Expo Go", "Android emulator", "iOS simulator"):
        assert _is_boilerplate(term), f"{term!r} should be flagged as boilerplate"


def test_package_manager_command_lists_are_boilerplate():
    assert _is_boilerplate("npm run dev\n# or\nyarn dev\n# or\npnpm dev\n# or\nbun dev")
    assert _is_boilerplate("Install dependencies with npm install before you start.")


def test_known_scaffolding_phrases_are_boilerplate():
    assert _is_boilerplate("This project was bootstrapped with create-next-app.")
    assert _is_boilerplate("Welcome to your Expo app - this is an Expo project.")


def test_genuine_project_specific_text_is_not_boilerplate():
    assert not _is_boilerplate(
        "Implemented an offline-first sync engine using WatermelonDB that "
        "reconciles conflicting edits from three devices in under 200ms."
    )
    assert not _is_boilerplate("Senior backend engineer with five years of experience.")
    assert not _is_boilerplate(None)
    assert not _is_boilerplate("")


# Real (reconstructed) default READMEs, as `create-next-app`/`create-expo-app`
# actually generate them, each with one genuine project-specific sentence
# appended - exactly the mixed-content scenario from the bug report.

NEXT_JS_BOILERPLATE_README = (
    "This is a Next.js project bootstrapped with create-next-app.\n\n"
    "## Getting Started\n\n"
    "First, run the development server:\n\n"
    "```bash\n"
    "npm run dev\n"
    "# or\n"
    "yarn dev\n"
    "# or\n"
    "pnpm dev\n"
    "# or\n"
    "bun dev\n"
    "```\n\n"
    "Open [http://localhost:3000](http://localhost:3000) with your browser to "
    "see the result.\n\n"
    "You can start editing the page by modifying `app/page.tsx`. The page "
    "auto-updates as you edit the file.\n\n"
    "## Learn More\n\n"
    "To learn more about Next.js, take a look at the following resources:\n\n"
    "- [Next.js Documentation](https://nextjs.org/docs)\n"
    "- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js "
    "tutorial.\n\n"
    "## Deploy on Vercel\n\n"
    "The easiest way to deploy your Next.js app is to use the Vercel Platform "
    "from the creators of Next.js.\n\n"
    "## What I built\n\n"
    "Implemented a real-time collaborative cursor overlay using WebSockets "
    "that syncs cursor positions across clients with sub-100ms latency."
)

EXPO_BOILERPLATE_README = (
    "# Welcome to your Expo app\n\n"
    "This is an [Expo](https://expo.dev) project created with "
    "[`create-expo-app`](https://www.npmjs.com/package/create-expo-app).\n\n"
    "## Get started\n\n"
    "1. Install dependencies\n\n"
    "   ```bash\n"
    "   npm install\n"
    "   ```\n\n"
    "2. Start the app\n\n"
    "   ```bash\n"
    "   npx expo start\n"
    "   ```\n\n"
    "In the output, you'll find options to open the app in a development "
    "build, Android emulator, iOS simulator, or Expo Go.\n\n"
    "## Get a fresh project\n\n"
    "When you're ready, run:\n\n"
    "```bash\n"
    "npm run reset-project\n"
    "```\n\n"
    "This command will move the starter code to the app-example directory.\n\n"
    "## Learn more\n\n"
    "To learn more about developing your project with Expo, look at the "
    "following resources.\n\n"
    "## Join the community\n\n"
    "Join our community of developers creating universal apps.\n\n"
    "## What I built\n\n"
    "Implemented an offline-first sync engine using WatermelonDB that "
    "reconciles conflicting edits from three devices in under 200ms."
)


def _claims_body(
    boilerplate_items: list[tuple[str, str]], genuine_claim_text: str, genuine_quote: str
) -> str:
    """
    A fake model response shaped exactly like the reported bug: several
    boilerplate items (the kind the old extractor was actually publishing)
    alongside one real, project-specific claim in the same batch. Every
    evidence_quote here is a real, sufficiently long, verbatim substring of
    the corresponding README fixture, and every claim_text shares vocabulary
    with its own quote - so the ONLY thing that can be excluding the
    boilerplate items is the new filter, not the grounding or overlap checks
    that already exist.
    """
    boilerplate_claims = [
        {"claim_text": claim_text, "skill_ids": [], "evidence_quote": quote}
        for claim_text, quote in boilerplate_items
    ]
    genuine_claim = {
        "claim_text": genuine_claim_text,
        "skill_ids": ["websockets"],
        "evidence_quote": genuine_quote,
    }
    return json.dumps({"claims": [*boilerplate_claims, genuine_claim]})


def test_nextjs_scaffolding_boilerplate_produces_no_claims_but_real_work_does():
    body = _claims_body(
        boilerplate_items=[
            ("yarn", "yarn dev\n# or\npnpm dev"),
            ("bootstrapped with create-next-app", "bootstrapped with create-next-app"),
            ("runs npm run dev to start", "npm run dev"),
        ],
        genuine_claim_text="built a real-time collaborative cursor overlay using WebSockets",
        genuine_quote="real-time collaborative cursor overlay using WebSockets",
    )
    claims = extract_candidate_claims(
        _StaticClient(body),
        document_id="doc-1",
        text=NEXT_JS_BOILERPLATE_README,
        source_type=SourceType.GITHUB_REPO,
    )

    # every boilerplate item excluded from all three passes, and the one
    # genuine claim each pass independently found collapses via dedup
    assert len(claims) == 1
    assert all(c.publishable for c in claims)
    assert all("WebSockets" in c.source_span.text for c in claims)
    assert not any(
        term in c.claim_text.lower() for c in claims for term in ("yarn", "npm", "create-next-app")
    )


def test_expo_scaffolding_boilerplate_produces_no_claims_but_real_work_does():
    body = _claims_body(
        boilerplate_items=[
            ("Android emulator", "Android emulator, iOS simulator, or Expo Go"),
            ("create-expo-app", "create-expo-app"),
            ("installs dependencies via npm install", "npm install"),
        ],
        genuine_claim_text="built an offline-first sync engine using WatermelonDB",
        genuine_quote="offline-first sync engine using WatermelonDB",
    )
    claims = extract_candidate_claims(
        _StaticClient(body),
        document_id="doc-1",
        text=EXPO_BOILERPLATE_README,
        source_type=SourceType.GITHUB_REPO,
    )

    assert len(claims) == 1
    assert all(c.publishable for c in claims)
    assert all("WatermelonDB" in c.source_span.text for c in claims)
    assert not any(
        term in c.claim_text.lower() for c in claims for term in ("emulator", "create-expo-app", "npm")
    )


# --- Tier assignment: rules-based, never model-supplied ---------------------
#
# _ExtractedSpan is the entire shape a model's response is parsed into. If it
# has no tier-like field, there is nothing for a model to leak into
# evidence_tier - the lookup below is the only thing that ever sets it.


def test_extracted_span_schema_has_no_tier_field():
    assert "evidence_tier" not in _ExtractedSpan.model_fields
    assert "tier" not in _ExtractedSpan.model_fields


def test_provisional_tier_is_deterministic_for_the_same_inputs():
    results = {_provisional_tier(SourceType.GITHUB_REPO, "history", "some text") for _ in range(5)}
    assert results == {EvidenceTier.T2}


def test_provisional_tier_github_is_always_t2_regardless_of_pass():
    for pass_name in ("identity", "history", "skills"):
        assert _provisional_tier(SourceType.GITHUB_REPO, pass_name, "anything") == EvidenceTier.T2


def test_provisional_tier_cv_skills_pass_is_self_declared():
    assert _provisional_tier(SourceType.CV, "skills", "Python, SQL, Docker") == EvidenceTier.T8


def test_provisional_tier_cv_history_pass_is_employer_confirmed():
    assert _provisional_tier(SourceType.CV, "history", "Led a team of 5 engineers") == EvidenceTier.T6


def test_provisional_tier_upwork_defaults_to_self_declared():
    text = "I saved a client 10 hours a week"
    assert _provisional_tier(SourceType.UPWORK_TEXT, "history", text) == EvidenceTier.T8


def test_provisional_tier_upwork_review_text_is_client_verified():
    text = "5-star review: client said the automation saved 10 hours a week"
    assert _provisional_tier(SourceType.UPWORK_TEXT, "history", text) == EvidenceTier.T1


_MULTI_FRAGMENT_SOURCE = (
    "Senior backend engineer. Shipped a fraud-detection model to production. "
    "Proficient in Python and Go."
)


def _single_claim_body(claim_text: str, quote: str) -> str:
    return json.dumps({"claims": [{"claim_text": claim_text, "skill_ids": [], "evidence_quote": quote}]})


def test_cv_pass_specific_tiers_are_assigned_without_llm_involvement():
    """
    Three distinct claims - one per pass, each grounded against a different
    fragment of the source - proving tier is decided purely by which pass
    produced the claim, never by the model. Each pass must return its OWN
    claim (not the same body echoed three times, like most other tests in
    this file use) precisely because these three claims are genuinely
    different accomplishments, not duplicates for the new dedup step to
    collapse - a fair test now that identical-body-per-pass would legitimately
    trigger deduplication instead of exercising tier assignment.
    """
    client = _PerPassClient(
        {
            "identity": _single_claim_body("senior backend engineer", "Senior backend engineer"),
            "history": _single_claim_body(
                "shipped a fraud detection model", "Shipped a fraud-detection model to production"
            ),
            "skills": _single_claim_body("proficient in python and go", "Proficient in Python and Go"),
        }
    )
    claims = extract_candidate_claims(
        client, document_id="doc-1", text=_MULTI_FRAGMENT_SOURCE, source_type=SourceType.CV
    )
    assert len(claims) == 3  # three distinct accomplishments, not duplicates
    tiers = sorted(c.evidence_tier.value for c in claims)
    # skills pass -> T8, identity + history passes -> T6
    assert tiers == ["T6", "T6", "T8"]


def test_ungrounded_claims_are_always_t8_regardless_of_source_type():
    body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "invented number",
                    "skill_ids": [],
                    "evidence_quote": "a paraphrase that does not appear verbatim",
                }
            ]
        }
    )
    claims = extract_candidate_claims(
        _StaticClient(body), document_id="doc-1", text=SOURCE, source_type=SourceType.GITHUB_REPO
    )
    assert all(c.evidence_tier == EvidenceTier.T8 for c in claims)


# --- Recency: observed_date and recency_factor are plumbed, not inert -------


def test_observed_date_and_recency_factor_flow_into_every_claim():
    body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "cut false positives",
                    "skill_ids": [],
                    "evidence_quote": "cutting false positives by 22%",
                }
            ]
        }
    )
    old_date = date(2019, 1, 1)
    claims = extract_candidate_claims(
        _StaticClient(body),
        document_id="doc-1",
        text=SOURCE,
        source_type=SourceType.GITHUB_REPO,
        observed_date=old_date,
    )
    assert claims
    assert all(c.publishable for c in claims)  # grounded, so recency isn't masked by T8 fallback
    assert all(c.observed_date == old_date for c in claims)
    assert all(c.recency_factor < 0.3 for c in claims)
    assert all(c.effective_weight < c.weight for c in claims)


def test_no_observed_date_means_no_recency_discount():
    body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "cut false positives",
                    "skill_ids": [],
                    "evidence_quote": "cutting false positives by 22%",
                }
            ]
        }
    )
    claims = extract_candidate_claims(
        _StaticClient(body), document_id="doc-1", text=SOURCE, source_type=SourceType.GITHUB_REPO
    )
    assert claims
    assert all(c.observed_date is None for c in claims)
    assert all(c.recency_factor == 1.0 for c in claims)
    assert all(c.effective_weight == c.weight for c in claims)


# --- Groq: a second provider behind the same LLMClient contract -------------
#
# GroqClient talks to Groq's OpenAI-compatible REST endpoint instead of the
# Anthropic SDK, but extract_candidate_claims never sees the difference - it
# only ever calls .complete(system=..., prompt=...). These tests exercise the
# real GroqClient class (not a hand-rolled fake) against a fake `requests`
# session, proving the same grounding and retry logic applies regardless of
# which provider answered.


class _FakeGroqResponse:
    def __init__(self, content: str, status_code: int = 200, headers: dict | None = None) -> None:
        self._content = content
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        # Groq's OpenAI-compatible chat-completions response shape.
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeGroqSession:
    """Returns each queued content string in order, one per call - shared
    across the extractor's three concurrent passes (identity/history/skills)."""

    def __init__(self, contents: list[str]) -> None:
        self._lock = threading.Lock()
        self._contents = list(contents)
        self.calls = 0

    def post(self, url, headers=None, json=None, timeout=None):
        with self._lock:
            content = self._contents[self.calls % len(self._contents)]
            self.calls += 1
        return _FakeGroqResponse(content)


_EMPTY_CLAIMS_BODY = json.dumps({"claims": []})


class _FlakyGroqSession:
    """Fails JSON-decodable content on exactly the first call across all
    threads, then returns a valid empty-claims body - mirrors _FlakyOnceClient
    above but exercises the real GroqClient parsing path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self.calls = 0

    def post(self, url, headers=None, json=None, timeout=None):
        with self._lock:
            self._count += 1
            self.calls += 1
            first = self._count == 1
        return _FakeGroqResponse("not json" if first else _EMPTY_CLAIMS_BODY)


def test_groq_client_parses_the_openai_compatible_response_shape():
    session = _FakeGroqSession(['{"claims": []}'])
    client = GroqClient(api_key="test-key", session=session)
    result = client.complete(system="sys", prompt="text")
    assert result == '{"claims": []}'
    assert session.calls == 1


def test_groq_client_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        GroqClient(api_key=None, session=_FakeGroqSession(["{}"]))


def test_groq_backed_extraction_grounds_claims_same_as_anthropic():
    body = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "cut false positives 22%",
                    "skill_ids": ["fraud_detection"],
                    "evidence_quote": "cutting false positives by 22%",
                }
            ]
        }
    )
    client = GroqClient(api_key="test-key", session=_FakeGroqSession([body]))
    claims = extract_candidate_claims(
        client, document_id="doc-1", text=SOURCE, source_type=SourceType.GITHUB_REPO
    )

    assert len(claims) == 1  # one per pass collapsed by dedup, same as Anthropic-backed
    assert all(c.publishable for c in claims)
    assert all(c.source_span.text == "cutting false positives by 22%" for c in claims)


def test_groq_backed_extraction_retries_on_malformed_response():
    client = GroqClient(api_key="test-key", session=_FlakyGroqSession())
    claims = extract_candidate_claims(
        client, document_id="doc-1", text=SOURCE, source_type=SourceType.CV
    )
    assert claims == []  # the recovered response is an empty claims list


class _RateLimitedThenOkSession:
    """Returns 429 for the first `fail_count` calls, then a valid body -
    stands in for Groq's free-tier requests-per-minute limit tripping during
    the three-concurrent-passes burst, then clearing."""

    def __init__(self, fail_count: int, body: str, retry_after: str | None = None) -> None:
        self._lock = threading.Lock()
        self._fail_count = fail_count
        self._body = body
        self._retry_after = retry_after
        self.calls = 0

    def post(self, url, headers=None, json=None, timeout=None):
        with self._lock:
            self.calls += 1
            call_number = self.calls
        if call_number <= self._fail_count:
            headers_out = {"Retry-After": self._retry_after} if self._retry_after else {}
            return _FakeGroqResponse("", status_code=429, headers=headers_out)
        return _FakeGroqResponse(self._body)


class _AlwaysRateLimitedSession:
    def post(self, url, headers=None, json=None, timeout=None):
        return _FakeGroqResponse("", status_code=429)


def test_groq_client_retries_on_429_and_eventually_succeeds():
    session = _RateLimitedThenOkSession(fail_count=2, body='{"claims": []}')
    sleeps: list[float] = []
    client = GroqClient(api_key="test-key", session=session, sleep=sleeps.append)

    result = client.complete(system="sys", prompt="text")

    assert result == '{"claims": []}'
    assert session.calls == 3
    assert len(sleeps) == 2  # slept once per 429 before the eventual success


def test_groq_client_honors_the_retry_after_header():
    session = _RateLimitedThenOkSession(fail_count=1, body="{}", retry_after="7")
    sleeps: list[float] = []
    client = GroqClient(api_key="test-key", session=session, sleep=sleeps.append)

    client.complete(system="sys", prompt="text")

    assert sleeps == [7.0]


def test_rate_limit_backoff_surfaces_through_status_reporting_across_threads():
    """
    The real thing this mechanism exists for: extract_candidate_claims runs
    three passes concurrently in a ThreadPoolExecutor, and status_reporting's
    contextvar has to survive that - just being set on the calling thread
    isn't enough proof, since ThreadPoolExecutor does NOT copy contextvars
    into worker threads on its own (same caveat _source_label's own comment
    calls out). Runs the full extract_candidate_claims path, not just a bare
    GroqClient.complete() call, so this actually exercises the copy_context()
    propagation into worker threads instead of just the main thread.
    """
    session = _RateLimitedThenOkSession(fail_count=2, body='{"claims": []}', retry_after="3")
    client = GroqClient(api_key="test-key", session=session, sleep=lambda _: None)

    events: list[dict] = []
    with status_reporting(lambda **fields: events.append(fields)):
        extract_candidate_claims(
            client, document_id="doc-1", text=SOURCE, source_type=SourceType.CV
        )

    rate_limit_events = [e for e in events if "waiting" in e.get("detail", "")]
    assert rate_limit_events == [{"detail": "waiting 3s (API rate limit)"}] * 2


def test_groq_client_gives_up_after_max_attempts_and_raises():
    sleeps: list[float] = []
    client = GroqClient(
        api_key="test-key", session=_AlwaysRateLimitedSession(), sleep=sleeps.append
    )

    with pytest.raises(requests.HTTPError):
        client.complete(system="sys", prompt="text")

    # one fewer sleep than attempts - the last attempt raises instead of retrying
    assert len(sleeps) == extractor.GROQ_MAX_ATTEMPTS - 1


def test_groq_client_logs_rate_limit_headers_when_finally_giving_up(caplog):
    """A bare '429 Client Error' tells you nothing about WHEN to retry - Groq
    sends Retry-After / x-ratelimit-reset-* headers saying exactly that, so
    the final failure must surface them (in the log, since the raised
    exception type stays requests.HTTPError - see the test above)."""

    class _RateLimitedWithHeaders:
        def post(self, url, headers=None, json=None, timeout=None):
            return _FakeGroqResponse(
                "", status_code=429, headers={"x-ratelimit-reset-requests": "2m59.56s"}
            )

    client = GroqClient(api_key="test-key", session=_RateLimitedWithHeaders(), sleep=lambda _: None)

    with caplog.at_level("ERROR"):
        with pytest.raises(requests.HTTPError):
            client.complete(system="sys", prompt="text")

    assert any("2m59.56s" in record.message for record in caplog.records)


# --- Provider selection: build_default_client() ------------------------------


class _SentinelAnthropicClient:
    pass


class _SentinelGroqClient:
    pass


def test_build_default_client_honors_explicit_llm_provider(monkeypatch):
    monkeypatch.setattr(extractor, "AnthropicClient", _SentinelAnthropicClient)
    monkeypatch.setattr(extractor, "GroqClient", _SentinelGroqClient)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert isinstance(build_default_client(), _SentinelGroqClient)

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(build_default_client(), _SentinelAnthropicClient)


def test_build_default_client_rejects_an_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError):
        build_default_client()


def test_build_default_client_defaults_to_whichever_key_is_present(monkeypatch):
    monkeypatch.setattr(extractor, "AnthropicClient", _SentinelAnthropicClient)
    monkeypatch.setattr(extractor, "GroqClient", _SentinelGroqClient)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "present")
    assert isinstance(build_default_client(), _SentinelGroqClient)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    assert isinstance(build_default_client(), _SentinelAnthropicClient)


# --- Startup validation: validate_llm_config() --------------------------------
#
# build_default_client() itself doesn't fail until the client actually makes
# its first real call (the Anthropic SDK doesn't validate a missing key at
# construction, only on first use) - so a misconfigured deploy would start
# fine and only fail on a real user's first analyze request. These tests
# cover both directions per env var/provider combination: the missing-key
# case actually raises, and the present-key case does NOT raise (so this
# can't be satisfied by a validator that just always raises).


def test_validate_llm_config_raises_naming_the_exact_missing_var_for_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        validate_llm_config()


def test_validate_llm_config_passes_when_anthropic_key_is_present(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-not-real")

    validate_llm_config()  # must not raise


def test_validate_llm_config_raises_naming_the_exact_missing_var_for_groq(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        validate_llm_config()


def test_validate_llm_config_passes_when_groq_key_is_present(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake-not-real")

    validate_llm_config()  # must not raise


def test_validate_llm_config_treats_a_blank_key_the_same_as_a_missing_one(monkeypatch):
    """Render's dashboard lets you create an env var with an empty value -
    that must fail exactly like the var being absent entirely, not pass a
    falsy-but-present check."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "   ")

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        validate_llm_config()


def test_validate_llm_config_rejects_an_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(RuntimeError, match="openai"):
        validate_llm_config()


def test_validate_llm_config_raises_when_no_provider_and_no_keys_at_all(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="No LLM credentials"):
        validate_llm_config()


def test_validate_llm_config_passes_with_no_provider_but_one_key_present(monkeypatch):
    """Matches build_default_client's own fallback exactly: no LLM_PROVIDER
    set falls back to whichever key exists."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake-not-real")

    validate_llm_config()  # must not raise


# --- Regression: grounded spans must actually relate to their claim --------
#
# reverify_span only proves internal consistency: the stored span.text still
# matches the source at ITS OWN indices. It cannot catch a claim being
# grounded at the wrong location, because a wrong-but-internally-consistent
# span reverifies just fine. This is exactly the bug that motivated the
# evidence_quote redesign: the model used to be asked to compute
# start_index/end_index directly, and its arithmetic was simply wrong - e.g.
# a claim about "Clerk" grounded to the unrelated fragment "st 20", and on a
# real 7898-character README, per-claim offset error (actual position minus
# model-reported start_index) ranged from -2393 to +3193 characters with no
# consistent pattern - every one of those spans still would have reverified.
#
# A hand-written fake LLMClient can't reproduce that failure mode (it only
# ever returns exactly what the test tells it to), so this necessarily
# exercises a REAL provider against real, substantial, known documents -
# skipped without a live key, like the real-Docling tests. Marked BOTH slow
# and live_api (not just slow): this doesn't merely take time, it spends real
# API quota, and a run under rate-limit backoff has been observed to take
# ~26 minutes rather than seconds. Run deliberately with
# `pytest -m live_api` before a release or after touching extraction/
# grounding logic - never as part of routine iteration. See the testing
# conventions note in CLAUDE.md.

_KNOWN_DOCUMENTS = {
    SourceType.CV: (
        "Jason Dhaki - Backend Engineer\n\n"
        "Senior backend engineer with five years of experience building "
        "distributed systems. At Acme Corp, built a real-time fraud "
        "detection pipeline that cut false positive alerts by 22 percent "
        "within the first quarter. Led a team of four engineers through a "
        "payments infrastructure rewrite that migrated the platform from a "
        "monolith to event-driven microservices. Proficient in Python, Go, "
        "PostgreSQL, and Kafka."
    ),
    SourceType.GITHUB_REPO: (
        "rag-pipeline\n\n"
        "Retrieval-augmented generation pipeline for legal document search.\n\n"
        "## What this does\n\n"
        "Indexes 100k legal documents into a vector store and serves "
        "grounded answers over a REST API. Rewrote the chunking strategy to "
        "split on section boundaries instead of fixed-size windows, which "
        "cut retrieval latency by 40 percent in production. Includes a "
        "FastAPI service, a background reindexing worker, and a small eval "
        "harness that scores retrieval precision against a held-out query set."
    ),
}


@pytest.mark.slow
@pytest.mark.live_api
@pytest.mark.skipif(
    not (os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    reason="requires a real LLM provider key (GROQ_API_KEY or ANTHROPIC_API_KEY)",
)
@pytest.mark.parametrize("source_type", [SourceType.CV, SourceType.GITHUB_REPO])
def test_grounded_spans_actually_relate_to_their_claim_on_real_documents(source_type):
    """
    For each claim a real model extracts from a real, substantial document,
    the grounded span must share at least one significant word with the
    claim_text describing it. reverify_span would pass even on the old
    index-guessing design's garbage spans, because that span.text really was
    sliced from the real document at those indices - it just wasn't where the
    claim actually lives. This test would have failed against that design
    (verified empirically before the fix) and passes against the
    evidence_quote design, because a verbatim quote that supports a claim
    necessarily shares vocabulary with the claim describing it.
    """
    text = _KNOWN_DOCUMENTS[source_type]
    client = build_default_client()
    claims = extract_candidate_claims(
        client, document_id="regression-doc", text=text, source_type=source_type
    )

    grounded = [c for c in claims if c.source_span is not None]
    assert grounded, "expected at least one claim to ground against a real, substantial document"

    for claim in grounded:
        claim_words = _significant_words(claim.claim_text)
        span_words = _significant_words(claim.source_span.text)
        assert claim_words & span_words, (
            f"claim {claim.claim_text!r} grounded to unrelated span "
            f"{claim.source_span.text!r} - no shared vocabulary"
        )
