"""
Tests for the seven pure scoring dimension functions (spec section 5). Each
function is (claims, benchmark) -> float, deterministic, no I/O - tested here
with hand-built Claim/Benchmark fixtures, no live API or real ingestion
needed for this layer.
"""

from __future__ import annotations

import pytest

from app.schemas import Benchmark, Claim, EvidenceTier, RateBand, SourceSpan, SourceType
from app.scoring.dimensions import (
    keyword_term_status,
    score_completeness,
    score_conversion,
    score_evidence_quality,
    score_keyword_coverage,
    score_portfolio_quality,
    score_positioning,
    score_pricing_strategy,
)


def _benchmark(**overrides) -> Benchmark:
    defaults = dict(
        niche="Test Niche",
        version="2026-01",
        required_terms=["n8n", "Make.com", "API integration", "webhook"],
        benchmark_topics=["workflow automation", "vector database"],
        title_formula="[role] - [vertical] - [outcome]",
        overview_words_min=280,
        overview_words_max=420,
        portfolio_min_items=3,
        portfolio_min_quantified=2,
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


def _span(text: str, document_id: str = "doc-1") -> SourceSpan:
    return SourceSpan(document_id=document_id, start_index=0, end_index=len(text), text=text)


def _claim(
    claim_text: str,
    source_type: SourceType,
    skill_ids: list[str] | None = None,
    span_text: str | None = None,
    document_id: str = "doc-1",
    tier: EvidenceTier = EvidenceTier.T2,
    weight: float = 0.85,
) -> Claim:
    span = _span(span_text, document_id) if span_text is not None else None
    return Claim(
        claim_text=claim_text,
        skill_ids=skill_ids or [],
        source_type=source_type,
        source_span=span,
        evidence_tier=tier,
        weight=weight,
    )


# --- positioning --------------------------------------------------------------


def test_positioning_scores_100_when_all_three_signals_present():
    claim = _claim(
        "Senior automation engineer built an n8n workflow automation that cut costs by 40%",
        SourceType.GITHUB_REPO,
        span_text="Senior automation engineer built an n8n workflow automation that cut costs by 40%",
    )
    assert score_positioning([claim], _benchmark()) == pytest.approx(100.0)


def test_positioning_scores_0_with_no_claims():
    assert score_positioning([], _benchmark()) == 0.0


def test_positioning_scores_partial_with_only_one_signal():
    # no role word, no niche-term overlap, but a number is present
    claim = _claim("Reduced costs by 40% for a client", SourceType.CV, span_text="Reduced costs by 40% for a client")
    assert score_positioning([claim], _benchmark()) == pytest.approx((1 / 3) * 100)


# --- evidence_quality -----------------------------------------------------------


def test_evidence_quality_takes_the_strongest_evidence_per_skill_not_the_sum():
    weak = _claim("uses n8n a little", SourceType.CV, skill_ids=["n8n"], span_text="uses n8n", tier=EvidenceTier.T6, weight=0.50)
    strong = _claim("shipped n8n workflow", SourceType.GITHUB_REPO, skill_ids=["n8n"], span_text="shipped n8n workflow", tier=EvidenceTier.T2, weight=0.85)
    # both claims are about the SAME skill - only the strongest should count,
    # not (0.50 + 0.85), which would exceed what a single T2 claim implies
    assert score_evidence_quality([weak, strong], _benchmark()) == pytest.approx(85.0)


def test_evidence_quality_ignores_ungrounded_claims():
    ungrounded = _claim("invented a skill", SourceType.CV, skill_ids=["python"], span_text=None, tier=EvidenceTier.T8, weight=0.15)
    assert score_evidence_quality([ungrounded], _benchmark()) == 0.0


def test_evidence_quality_ignores_claims_with_no_skill_ids():
    claim = _claim("shipped something", SourceType.GITHUB_REPO, skill_ids=[], span_text="shipped something")
    assert score_evidence_quality([claim], _benchmark()) == 0.0


def test_evidence_quality_is_0_with_no_claims():
    assert score_evidence_quality([], _benchmark()) == 0.0


# --- keyword_coverage ------------------------------------------------------------


def test_keyword_coverage_full_marks_when_everything_present():
    text = (
        "Built n8n and Make.com workflows with API integration and webhook triggers. "
        "This covers workflow automation and vector database patterns."
    )
    claim = _claim(text, SourceType.GITHUB_REPO, span_text=text)
    assert score_keyword_coverage([claim], _benchmark()) == pytest.approx(100.0)


def test_keyword_coverage_hard_cap_on_repetition():
    """A term present once scores identically to one present nine times -
    presence, never frequency."""
    once = _claim("uses n8n", SourceType.GITHUB_REPO, span_text="uses n8n")
    nine_times = _claim(
        "n8n n8n n8n n8n n8n n8n n8n n8n n8n", SourceType.GITHUB_REPO, span_text="n8n n8n n8n n8n n8n n8n n8n n8n n8n"
    )
    benchmark = _benchmark(required_terms=["n8n"], benchmark_topics=[])
    assert score_keyword_coverage([once], benchmark) == score_keyword_coverage([nine_times], benchmark)


def test_keyword_coverage_semantic_half_matches_via_synonym():
    """'Pinecone' should count toward the 'vector database' topic - the
    explicit synonym-map placeholder for real embeddings."""
    claim = _claim("Used Pinecone for retrieval", SourceType.GITHUB_REPO, span_text="Used Pinecone for retrieval")
    benchmark = _benchmark(required_terms=[], benchmark_topics=["vector database"])
    assert score_keyword_coverage([claim], benchmark) == pytest.approx(100.0)


def test_keyword_coverage_is_0_when_nothing_matches():
    claim = _claim("completely unrelated text", SourceType.CV, span_text="completely unrelated text")
    assert score_keyword_coverage([claim], _benchmark()) == 0.0


def test_keyword_coverage_defaults_to_full_marks_when_benchmark_has_no_terms():
    benchmark = _benchmark(required_terms=[], benchmark_topics=[])
    assert score_keyword_coverage([], benchmark) == pytest.approx(100.0)


def test_keyword_term_status_reports_presence_per_term_not_a_rolled_up_count():
    text = "Built n8n workflows with webhook triggers."
    claim = _claim(text, SourceType.GITHUB_REPO, span_text=text)
    benchmark = _benchmark(required_terms=["n8n", "Make.com", "webhook"], benchmark_topics=[])

    status = keyword_term_status([claim], benchmark)

    assert status == [
        {"term": "n8n", "present": True},
        {"term": "Make.com", "present": False},
        {"term": "webhook", "present": True},
    ]


def test_keyword_term_status_matches_score_keyword_coverage_required_half():
    """The per-term present count here must agree with what
    score_keyword_coverage's own required-term half computed - this is the
    same _contains check, surfaced instead of collapsed into one number."""
    text = "Built n8n workflows."
    claim = _claim(text, SourceType.GITHUB_REPO, span_text=text)
    benchmark = _benchmark(required_terms=["n8n", "Make.com"], benchmark_topics=[])

    status = keyword_term_status([claim], benchmark)
    present_count = sum(1 for item in status if item["present"])

    assert present_count == 1


# --- portfolio_quality -----------------------------------------------------------


def test_portfolio_quality_counts_distinct_grounded_project_items():
    items = [
        _claim("project one", SourceType.GITHUB_REPO, span_text="project one shipped", document_id="doc-1"),
        _claim("project two", SourceType.PORTFOLIO_SITE, span_text="project two shipped", document_id="doc-2"),
        _claim("project three cut costs 40%", SourceType.GITHUB_REPO, span_text="project three cut costs 40%", document_id="doc-3"),
    ]
    benchmark = _benchmark(portfolio_min_items=3, portfolio_min_quantified=1)
    assert score_portfolio_quality(items, benchmark) == pytest.approx(100.0)


def test_portfolio_quality_ignores_non_project_and_ungrounded_claims():
    non_project = _claim("wrote an article", SourceType.ARTICLE, span_text="wrote an article")
    ungrounded_project = _claim("claims a repo", SourceType.GITHUB_REPO, span_text=None)
    assert score_portfolio_quality([non_project, ungrounded_project], _benchmark()) == 0.0


def test_portfolio_quality_partial_when_below_targets():
    one_item_unquantified = _claim("built a tool", SourceType.GITHUB_REPO, span_text="built a tool")
    benchmark = _benchmark(portfolio_min_items=2, portfolio_min_quantified=2)
    # item_score = 1/2, quantified_score = 0/2 -> average 0.25 -> 25.0
    assert score_portfolio_quality([one_item_unquantified], benchmark) == pytest.approx(25.0)


def test_portfolio_quality_counts_two_claims_sharing_an_identical_span_once():
    """
    Two different extraction passes can independently ground two DIFFERENT
    claims - real claims, not a grounding bug - to the exact same source
    sentence (e.g. a "skill used" claim and a "project demonstrated" claim
    both citing the same README line, as seen in a real run). That's one
    piece of portfolio evidence, not two: item_ids is keyed by
    source_span.document_id, so two claims sharing an identical span (same
    document_id, same start/end indices, since both are built from the same
    span_text/document_id here) collapse to the same set element regardless
    of how many distinct claims cite it.
    """
    shared_span_text = "Built the HeroScene with Three.js and dynamic imports for heavy assets"
    skill_claim = _claim(
        "Used Three.js", SourceType.GITHUB_REPO, span_text=shared_span_text, document_id="doc-1"
    )
    portfolio_claim = _claim(
        "Implemented dynamic imports for heavy assets, specifically the HeroScene",
        SourceType.GITHUB_REPO,
        span_text=shared_span_text,
        document_id="doc-1",
    )
    # portfolio_min_quantified=0 isolates item counting from the quantified
    # half of the score, which this case doesn't otherwise exercise.
    benchmark = _benchmark(portfolio_min_items=2, portfolio_min_quantified=0)

    # If this counted per-claim instead of per-span/document, two claims
    # would satisfy portfolio_min_items=2 outright (score 100.0). It must
    # instead score as exactly ONE item against a target of two: item_score
    # = 1/2 = 0.5, quantified_score fixed at 1.0 -> average 0.75 -> 75.0.
    assert score_portfolio_quality(
        [skill_claim, portfolio_claim], benchmark
    ) == pytest.approx(75.0)


# --- completeness ------------------------------------------------------------------


def test_completeness_full_marks_with_all_four_signals():
    claims = [
        _claim("work history", SourceType.CV, span_text="work history"),
        _claim("shipped project", SourceType.GITHUB_REPO, span_text="shipped project"),
        _claim("skills", SourceType.CV, skill_ids=["a", "b", "c"], span_text="skills"),
        _claim("self pitch", SourceType.UPWORK_TEXT, span_text="self pitch"),
    ]
    assert score_completeness(claims, _benchmark()) == pytest.approx(100.0)


def test_completeness_is_0_with_no_claims():
    assert score_completeness([], _benchmark()) == 0.0


def test_completeness_partial_with_only_some_checklist_items():
    claim = _claim("cv text", SourceType.CV, span_text="cv text")
    # has_identity_history=True, everything else False -> 1/4 = 25.0
    assert score_completeness([claim], _benchmark()) == pytest.approx(25.0)


# --- conversion --------------------------------------------------------------------


def test_conversion_scores_the_fraction_of_outcome_framed_claims():
    outcome = _claim("cut costs by 40%", SourceType.GITHUB_REPO, span_text="cut costs by 40%")
    self_description = _claim("I am a hard worker", SourceType.CV, span_text="I am a hard worker")
    assert score_conversion([outcome, self_description], _benchmark()) == pytest.approx(50.0)


def test_conversion_is_0_with_no_claims():
    assert score_conversion([], _benchmark()) == 0.0


# --- pricing_strategy --------------------------------------------------------------


def test_pricing_strategy_scores_share_of_claims_in_justifying_tiers():
    t1_claim = _claim("client outcome", SourceType.UPWORK_TEXT, span_text="client outcome", tier=EvidenceTier.T1, weight=1.0)
    t6_claim = _claim("work history", SourceType.CV, span_text="work history", tier=EvidenceTier.T6, weight=0.50)
    benchmark = _benchmark()  # justifying_tiers = ["T1", "T2"]
    assert score_pricing_strategy([t1_claim, t6_claim], benchmark) == pytest.approx(50.0)


def test_pricing_strategy_ignores_ungrounded_claims():
    ungrounded = _claim("invented", SourceType.CV, span_text=None, tier=EvidenceTier.T1, weight=1.0)
    assert score_pricing_strategy([ungrounded], _benchmark()) == 0.0


def test_pricing_strategy_is_0_with_no_grounded_claims():
    assert score_pricing_strategy([], _benchmark()) == 0.0
