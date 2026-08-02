"""
Tests for the generator + validator, built and tested together (spec section
8). The LLMClient boundary is the only thing faked - exactly the same
pattern app/ingestion/extractor.py's own tests use - so these run offline,
deterministically, with no live Groq/Anthropic calls anywhere.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from app.generation import AssetValidationError, generate_assets
from app.generation.generator import DraftAsset, generate_overview_draft
from app.generation.validator import validate_asset
from app.ingestion.span_grounding import ground_span
from app.schemas import Benchmark, Claim, EvidenceTier, RateBand, SourceType
from app.storage.store import file_store


def _benchmark(**overrides) -> Benchmark:
    defaults = dict(
        niche="SMB workflow automation",
        version="2026-01",
        required_terms=["n8n"],
        benchmark_topics=[],
        title_formula="[role] - [vertical] - [outcome]",
        overview_words_min=50,
        overview_words_max=200,
        portfolio_min_items=1,
        rate_band=RateBand(low=50.0, high=80.0, justifying_tiers=["T1", "T2"]),
        dimension_targets={
            "positioning": 85.0,
            "evidence_quality": 80.0,
            "keyword_coverage": 90.0,
            "portfolio_quality": 85.0,
            "completeness": 85.0,
            "conversion": 80.0,
            "pricing_strategy": 80.0,
        },
    )
    defaults.update(overrides)
    return Benchmark(**defaults)


def _stored_claim(
    text: str,
    source_type: SourceType = SourceType.GITHUB_REPO,
    tier: EvidenceTier = EvidenceTier.T2,
    weight: float = 0.85,
) -> Claim:
    """A claim grounded against a REAL stored document - not just a hand-
    built SourceSpan pointing at nothing - so file_store.get_text and
    reverify_span (which validate_asset actually calls) resolve correctly."""
    pair = file_store.put_source(original=text.encode("utf-8"), text=text)
    span = ground_span(document_id=pair.text_id, source_text=text, start_index=0, end_index=len(text))
    return Claim(
        claim_text=text,
        source_type=source_type,
        source_span=span,
        evidence_tier=tier,
        weight=weight,
    )


class _StaticClient:
    """Returns the same JSON body for every call - both the title and
    overview builders see it."""

    def __init__(self, body: str) -> None:
        self.body = body
        self.calls = 0

    def complete(self, *, system: str, prompt: str) -> str:
        self.calls += 1
        return self.body


# --- Structural guarantee: the generator cannot construct a validated asset --


def test_draft_asset_has_no_validated_field_to_fake():
    """The structural guarantee: DraftAsset cannot even REPRESENT a
    validated=True state - there is no field a generator bug could set."""
    field_names = {f.name for f in dataclasses.fields(DraftAsset)}
    assert "validated" not in field_names


def test_generator_module_has_no_way_to_construct_a_generated_asset():
    """generator.py works entirely in DraftAsset terms - it doesn't even
    import the real schema type, so there is no code path in this module
    that could produce one, validated or not."""
    import app.generation.generator as generator_module

    assert not hasattr(generator_module, "GeneratedAsset")


# --- Title: well-grounded generation passes end-to-end -----------------------


def test_title_generation_passes_validation_with_well_grounded_evidence():
    claim = _stored_claim("Cut retrieval latency by 40% for a legal-tech client")
    benchmark = _benchmark()
    body = json.dumps({"text": "RAG Engineer - Legal Tech - cut retrieval latency by 40%"})

    assets, _incomplete = generate_assets([claim], benchmark, client=_StaticClient(body))

    titles = [a for a in assets if a.kind == "title"]
    assert len(titles) == 1
    assert titles[0].validated is True
    assert titles[0].text == "RAG Engineer - Legal Tech - cut retrieval latency by 40%"
    assert titles[0].source_spans  # backing evidence is attached, not empty


def test_title_generation_returns_nothing_when_no_claim_has_a_numeric_outcome():
    claim = _stored_claim("Worked on a project")  # no number anywhere
    benchmark = _benchmark()
    body = json.dumps({"text": "should never be requested"})

    assets, incomplete = generate_assets([claim], benchmark, client=_StaticClient(body))

    assert [a for a in assets if a.kind == "title"] == []
    # nothing to draft from at all is an expected empty result, not a failure
    assert incomplete is False


# --- Title: tier-aware "verified" badge + proof-implying language guard -----
#
# Spec section 3: T8 is "self-declared, with no corroboration" - a claim can
# be perfectly well-grounded (a real, verbatim quote) while still being
# nothing but the user's own unverified sentence about themselves. A title
# reading "Proven" / "Verified" off ONLY that kind of evidence overclaims
# exactly what the evidence cap exists to prevent.


def test_title_asset_is_tier_verified_when_backed_only_by_proof_tier_evidence():
    claim = _stored_claim(
        "Cut retrieval latency by 40% for a legal-tech client", tier=EvidenceTier.T2, weight=0.85
    )
    benchmark = _benchmark()
    body = json.dumps({"text": "RAG Engineer - Legal Tech - cut retrieval latency by 40%"})

    assets, _incomplete = generate_assets([claim], benchmark, client=_StaticClient(body))

    titles = [a for a in assets if a.kind == "title"]
    assert len(titles) == 1
    assert titles[0].tier_verified is True


def test_title_asset_is_not_tier_verified_when_backed_only_by_self_declared_evidence():
    claim = _stored_claim("Cut costs by 40% for clients", tier=EvidenceTier.T8, weight=0.15)
    benchmark = _benchmark()
    body = json.dumps({"text": "Automation Specialist - SMB - cut costs by 40%"})

    assets, _incomplete = generate_assets([claim], benchmark, client=_StaticClient(body))

    titles = [a for a in assets if a.kind == "title"]
    assert len(titles) == 1
    assert titles[0].tier_verified is False


def test_validate_asset_rejects_proof_implying_language_backed_only_by_self_declared_evidence():
    weak_claim = _stored_claim("Cut costs by 40% for clients", tier=EvidenceTier.T8, weight=0.15)
    draft = DraftAsset(
        kind="title",
        text="Automation Specialist - Proven to cut costs by 40%",
        claim_refs=[weak_claim],
    )

    with pytest.raises(AssetValidationError, match="proof-implying"):
        validate_asset(draft)


def test_validate_asset_accepts_proof_implying_language_backed_by_proof_tier_evidence():
    strong_claim = _stored_claim("Cut costs by 40% for clients", tier=EvidenceTier.T2, weight=0.85)
    draft = DraftAsset(kind="title", text="Verified to cut costs by 40%", claim_refs=[strong_claim])

    asset = validate_asset(draft)

    assert asset.validated is True
    assert asset.tier_verified is True


def test_title_generation_with_t8_only_evidence_never_publishes_proof_implying_language():
    """End-to-end adversarial case: a T8-only claim set with a numeric
    outcome, and a client that ALWAYS answers with overclaiming language
    (simulating the model ignoring the prompt's own instruction not to). The
    validator's independent check must still catch and drop it - never
    silently publish 'Proven'/'Verified' backed by nothing but a self-
    declared sentence."""
    claim = _stored_claim("Cut costs by 40% for a client", tier=EvidenceTier.T8, weight=0.15)
    benchmark = _benchmark()

    class _AlwaysOverclaimsClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *, system: str, prompt: str) -> str:
            self.calls += 1
            return json.dumps({"text": "Automation Specialist - Proven 40% Cost Cuts"})

    client = _AlwaysOverclaimsClient()
    assets, incomplete = generate_assets([claim], benchmark, client=client)

    titles = [a for a in assets if a.kind == "title"]
    assert titles == []  # dropped, never published with proof-implying language
    assert client.calls == 2  # both retry attempts actually ran
    assert incomplete is True
    for asset in assets:
        text_lower = asset.text.lower()
        assert "proven" not in text_lower
        assert "verified" not in text_lower
        assert "guaranteed" not in text_lower


# --- Title: fabricated number is caught (the adversarial case) --------------


class _AlwaysFabricatesClient:
    """Always cites a number the given evidence never mentions - the same
    class of failure ingestion's evidence_quote fix caught: a plausible-
    sounding figure with nothing real behind it."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system: str, prompt: str) -> str:
        self.calls += 1
        return json.dumps({"text": "RAG Engineer - Legal Tech - cut retrieval latency by 95%"})


def test_fabricated_number_in_title_is_caught_and_dropped_after_retries_exhausted():
    # Tier T8 so the overview builder finds no proof-tier evidence and never
    # calls the client at all - isolates the call count to title attempts only.
    claim = _stored_claim(
        "Cut retrieval latency by 40% for a legal-tech client",
        tier=EvidenceTier.T8,
        weight=0.15,
    )
    benchmark = _benchmark()
    client = _AlwaysFabricatesClient()

    assets, incomplete = generate_assets([claim], benchmark, client=client)

    assert [a for a in assets if a.kind == "title"] == []  # dropped, never returned unvalidated
    assert client.calls == 2  # both attempts (RETRY_ATTEMPTS) actually ran, not just one
    # a title WAS attempted (real evidence existed) but never validated - this
    # must be surfaced, not indistinguishable from "nothing to generate"
    assert incomplete is True


class _FailsOnceThenGroundedClient:
    """Fabricates on the first call, then returns a properly grounded title
    on the retry - proves the "regenerate once" recovery path actually
    works, not just the give-up path."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system: str, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"text": "RAG Engineer - Legal Tech - cut retrieval latency by 95%"})
        return json.dumps({"text": "RAG Engineer - Legal Tech - cut retrieval latency by 40%"})


def test_fabricated_number_recovers_via_one_retry():
    claim = _stored_claim(
        "Cut retrieval latency by 40% for a legal-tech client",
        tier=EvidenceTier.T8,
        weight=0.15,
    )
    benchmark = _benchmark()
    client = _FailsOnceThenGroundedClient()

    assets, incomplete = generate_assets([claim], benchmark, client=client)

    titles = [a for a in assets if a.kind == "title"]
    assert len(titles) == 1
    assert titles[0].validated is True
    assert "40%" in titles[0].text
    assert client.calls == 2  # first attempt failed validation, second succeeded
    assert incomplete is False  # recovered on retry - nothing incomplete about it


# --- Overview: proof section drawn only from T1-T4 --------------------------


def test_overview_generation_passes_validation_with_only_proof_tier_evidence():
    claim = _stored_claim("Delivered a project that cut costs by 40% for a client")
    benchmark = _benchmark()
    body = json.dumps({"text": "We deliver measurable results, cutting costs by 40% for clients."})

    assets, _incomplete = generate_assets([claim], benchmark, client=_StaticClient(body))

    overviews = [a for a in assets if a.kind == "overview"]
    assert len(overviews) == 1
    assert overviews[0].validated is True


def test_generated_overview_excludes_t5_through_t8_claims_from_proof_even_when_grounded():
    """The explicitly adversarial case: a T8 claim IS grounded (a real
    stored span backs it), but its evidence must never surface as "proof" in
    a generated overview. Simulates the model citing it anyway (it was never
    shown this claim's text at all - see the generator-level test below -
    but this proves the validator ALSO independently refuses it, not only
    the generator's own filtering)."""
    strong_claim = _stored_claim("Delivered a project that cut costs by 40%", tier=EvidenceTier.T2, weight=0.85)
    weak_claim = _stored_claim("Self-declared: increased revenue by 99%", tier=EvidenceTier.T8, weight=0.15)
    benchmark = _benchmark()
    body = json.dumps({"text": "Delivered results, including a 99% revenue increase."})

    assets, incomplete = generate_assets([strong_claim, weak_claim], benchmark, client=_StaticClient(body))

    # "99%" only exists in the T8 claim's span - never in claim_refs for an
    # overview draft - so it fails the number-tracing check and is dropped.
    assert [a for a in assets if a.kind == "overview"] == []
    # the overview WAS attempted (T2 evidence existed to draft from) but
    # never validated - must be surfaced, not silently indistinguishable
    # from "nothing to generate". (Title still succeeds here, since a
    # title has no T1-T4 restriction and the T8 claim's own span DOES back
    # "99%" - only the overview's proof-tier rule rejects it.)
    assert incomplete is True


def test_overview_generator_never_shows_the_model_t5_through_t8_evidence():
    """Generator-level proof: the weak claim's text is never even in the
    prompt sent to the model - the T1-T4 filter happens before the model is
    ever called, not after, as a post-hoc check on what it wrote."""
    strong_claim = _stored_claim("Delivered a project that cut costs by 40%", tier=EvidenceTier.T2, weight=0.85)
    weak_claim = _stored_claim("I am a hard worker who never gives up", tier=EvidenceTier.T8, weight=0.15)

    class _RecordingClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, *, system: str, prompt: str) -> str:
            self.prompts.append(prompt)
            return json.dumps({"text": "Delivered results, cutting costs by 40% for clients."})

    client = _RecordingClient()
    draft = generate_overview_draft(client, [strong_claim, weak_claim], _benchmark())

    assert draft is not None
    assert weak_claim not in draft.claim_refs
    assert all("hard worker" not in prompt for prompt in client.prompts)


def test_validate_asset_rejects_an_overview_draft_referencing_a_t5_to_t8_claim():
    """Direct unit test of the validator's OWN independent tier check,
    bypassing the LLM/generator entirely - even if a future generator bug
    ever handed the model T5-T8 evidence for an overview, this must still
    refuse it."""
    weak_claim = _stored_claim("I am a hard worker", tier=EvidenceTier.T8, weight=0.15)
    draft = DraftAsset(kind="overview", text="I am a hard worker.", claim_refs=[weak_claim])

    with pytest.raises(AssetValidationError, match="below T4"):
        validate_asset(draft)


def test_overview_generation_returns_nothing_with_no_proof_tier_evidence_at_all():
    claim = _stored_claim("I am a hard worker", tier=EvidenceTier.T8, weight=0.15)
    benchmark = _benchmark()
    body = json.dumps({"text": "should never be requested"})

    assets, incomplete = generate_assets([claim], benchmark, client=_StaticClient(body))

    assert [a for a in assets if a.kind == "overview"] == []
    # no number anywhere and no T1-T4 evidence - nothing to draft from at
    # all (for either kind), not a failure
    assert incomplete is False


# --- The validator re-reads storage, never trusts a cached span -------------


def test_validate_asset_rejects_a_draft_whose_source_has_drifted_since_extraction():
    """If the underlying stored document changed after the claim was
    extracted, the span no longer reverifies against the CURRENT text - the
    draft must be rejected even though it looked fine when it was made."""
    claim = _stored_claim("Cut retrieval latency by 40%")
    draft = DraftAsset(kind="title", text="cut retrieval latency by 40%", claim_refs=[claim])

    # Simulate drift by overwriting the stored text file directly - bypasses
    # FileStore's own (intentionally immutable) API, same pattern
    # test_store.py already uses (`store._resolve(...)`) to reach the
    # underlying file for test setup.
    stored_path = file_store._resolve(claim.source_span.document_id)
    stored_path.write_text("completely different text now", encoding="utf-8")

    with pytest.raises(AssetValidationError, match="no longer reverifies"):
        validate_asset(draft)


def test_validate_asset_rejects_a_claim_with_no_source_span():
    unglued_claim = Claim(
        claim_text="invented",
        source_type=SourceType.CV,
        source_span=None,
        evidence_tier=EvidenceTier.T8,
        weight=0.15,
    )
    draft = DraftAsset(kind="title", text="invented", claim_refs=[unglued_claim])

    with pytest.raises(AssetValidationError, match="no source span"):
        validate_asset(draft)


# --- Number matching: word boundaries and unit context, not bare substrings -


def test_validate_asset_rejects_a_number_that_only_matches_by_digit_coincidence():
    """The exact reported bypass vector: '20' IS a substring of '2019', so a
    fabricated '20%' must NOT validate just because the span happens to
    mention an unrelated year range. The word-boundary requirement means
    '20' cannot match inside '2019' - there's no boundary between the '0'
    and the following '1', both being \\w characters."""
    claim = _stored_claim("The platform grew from 2019 to 2024 across three regions")
    draft = DraftAsset(kind="title", text="Cut costs by 20% for clients", claim_refs=[claim])

    with pytest.raises(AssetValidationError, match="not backed"):
        validate_asset(draft)


def test_validate_asset_still_accepts_a_genuine_word_bounded_number_beside_a_decoy_year():
    """No over-rejection regression: a real, standalone, word-bounded number
    must still validate even when the same span also happens to contain an
    unrelated year that could have produced a false positive under the old
    bare substring check (and, before the word-boundary fix, could just as
    easily have produced a false NEGATIVE by chance - digit coincidences cut
    both ways)."""
    claim = _stored_claim("Active from 2019 to 2024, the team cut onboarding time by 40%")
    draft = DraftAsset(kind="title", text="Cut onboarding time by 40%", claim_refs=[claim])

    asset = validate_asset(draft)

    assert asset.validated is True


def test_validate_asset_rejects_a_percentage_backed_only_by_a_bare_count():
    """Unit/context agreement: the span genuinely contains the digits '40'
    with a real word boundary, but only as a bare count ('40 clients'), never
    as a percentage. A generated '40%' must not be laundered through a span
    that never actually expresses that number as a percentage."""
    claim = _stored_claim("Served 40 clients across the region last year")
    draft = DraftAsset(kind="title", text="Trusted by 40% of the market", claim_refs=[claim])

    with pytest.raises(AssetValidationError, match="not backed"):
        validate_asset(draft)


def test_validate_asset_does_not_require_percent_context_for_a_bare_count_claim():
    """The unit-context check only applies when the GENERATED text itself
    presents the number as a percentage - a plain count in the draft only
    needs a plain, word-bounded match, not an invented percent sign."""
    claim = _stored_claim("Served 40 clients across the region last year")
    draft = DraftAsset(kind="title", text="Served 40 clients nationwide", claim_refs=[claim])

    asset = validate_asset(draft)

    assert asset.validated is True


def test_validate_asset_does_not_back_a_number_using_a_different_claims_span():
    """Per-claim scoping: a number attributed to the draft must be backed by
    ONE claim's own span, not assembled from fragments of different claims'
    spans concatenated together. Claim A has the digits with no percent
    context; claim B has an unrelated percentage - neither one, alone,
    actually backs '40%'."""
    claim_a = _stored_claim("Served 40 clients across the region last year")
    claim_b = _stored_claim("Renewed 15% of contracts early")
    draft = DraftAsset(kind="title", text="Trusted by 40% of the market", claim_refs=[claim_a, claim_b])

    with pytest.raises(AssetValidationError, match="not backed"):
        validate_asset(draft)


# --- generate_assets orchestration -------------------------------------------


def test_generate_assets_returns_only_validated_assets():
    claim = _stored_claim("Cut retrieval latency by 40% for a legal-tech client")
    benchmark = _benchmark()
    body = json.dumps({"text": "RAG Engineer - Legal Tech - cut retrieval latency by 40%"})

    assets, _incomplete = generate_assets([claim], benchmark, client=_StaticClient(body))

    assert assets  # at least one asset was produced
    assert all(a.validated for a in assets)


def test_generate_assets_returns_an_empty_list_with_no_claims_at_all():
    benchmark = _benchmark()
    body = json.dumps({"text": "should never be requested"})

    assets, incomplete = generate_assets([], benchmark, client=_StaticClient(body))
    assert assets == []
    assert incomplete is False  # nothing to draft from at all, not a failure
