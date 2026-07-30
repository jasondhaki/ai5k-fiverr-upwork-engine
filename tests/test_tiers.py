"""
Tests for the real, rules-based tier assignment (assign_tiers, spec section
3). Every rule in the table gets its own test, plus the two cross-cutting
behaviors that matter most: a stronger source_type never loses to a weaker
one just because a certification is mentioned in passing (T2 outranks T4),
and an unverifiable certification claim never keeps a stronger tier just
because of where it happened to be found.
"""

from __future__ import annotations

import pytest

from app.evidence.tiers import assign_tiers
from app.schemas import Claim, EvidenceTier, SourceSpan, SourceType


def _span(text: str, document_id: str = "doc-1") -> SourceSpan:
    return SourceSpan(document_id=document_id, start_index=0, end_index=len(text), text=text)


def _claim(
    claim_text: str,
    source_type: SourceType,
    span_text: str | None = None,
    provisional_tier: EvidenceTier = EvidenceTier.T8,
) -> Claim:
    """A claim as it would arrive at assign_tiers: already grounded (or not),
    already carrying whatever provisional tier ingestion assigned."""
    span = _span(span_text) if span_text is not None else None
    return Claim(
        claim_text=claim_text,
        source_type=source_type,
        source_span=span,
        evidence_tier=provisional_tier,
        weight=0.15,
    )


def _refined(claim: Claim) -> Claim:
    return assign_tiers([claim])[0]


# --- Rule: not_grounded -------------------------------------------------------


def test_ungrounded_claim_is_always_t8_regardless_of_text():
    claim = _claim(
        "Client said this cut onboarding time by 80%",  # otherwise T1-shaped text
        SourceType.UPWORK_TEXT,
        span_text=None,
        provisional_tier=EvidenceTier.T2,  # even a strong provisional tier
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T8
    assert result.tier_rule == "not_grounded"
    assert result.weight == pytest.approx(0.15)


# --- Rule: client_verified_outcome (T1) --------------------------------------


@pytest.mark.parametrize(
    "source_type",
    [SourceType.UPWORK_TEXT, SourceType.GITHUB_REPO, SourceType.CV, SourceType.PORTFOLIO_SITE],
)
def test_client_verified_outcome_is_t1_regardless_of_source_type(source_type):
    span_text = "Client said this cut their onboarding time by 80 percent"
    claim = _claim("cut onboarding time 80%", source_type, span_text=span_text)
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T1
    assert result.tier_rule == "client_verified_outcome"
    assert result.weight == pytest.approx(1.00)


def test_five_star_review_language_is_also_t1():
    claim = _claim(
        "delivered a 5-star result",
        SourceType.UPWORK_TEXT,
        span_text="5-star review: automation saved 10 hours a week",
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T1
    assert result.tier_rule == "client_verified_outcome"


# --- Rule: project_demonstrated (T2) -----------------------------------------


@pytest.mark.parametrize(
    "source_type",
    [
        SourceType.GITHUB_REPO,
        SourceType.PORTFOLIO_SITE,
        SourceType.HUGGINGFACE,
        SourceType.DEMO_VIDEO,
    ],
)
def test_project_demonstrated_sources_are_t2(source_type):
    claim = _claim("built a RAG pipeline", source_type, span_text="Built a RAG pipeline over 100k docs")
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T2
    assert result.tier_rule == "project_demonstrated"
    assert result.weight == pytest.approx(0.85)


def test_project_demonstrated_outranks_a_certification_mention_in_the_same_claim():
    """The T2-outranks-T4 ordering: a GitHub repo claim whose text ALSO
    contains a full, verifiable certification signal must still land on T2 -
    shipped work outranks a certification mention, never the reverse."""
    span_text = (
        "AWS Certified Solutions Architect, proctored exam, verify at "
        "aws.amazon.com/verification. Also shipped this deployed repo."
    )
    claim = _claim("shipped a deployed repo", SourceType.GITHUB_REPO, span_text=span_text)
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T2
    assert result.tier_rule == "project_demonstrated"


# --- Rule: platform_assessed (T3) --------------------------------------------


def test_platform_assessment_language_on_onboarding_form_is_t3():
    claim = _claim(
        "scored well on the skills assessment",
        SourceType.ONBOARDING_FORM,
        span_text="Completed our platform assessment: skills assessment score 92/100",
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T3
    assert result.tier_rule == "platform_assessed"


def test_onboarding_form_without_assessment_language_is_self_declared():
    claim = _claim(
        "five years of experience",
        SourceType.ONBOARDING_FORM,
        span_text="I have five years of experience in backend development",
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T8
    assert result.tier_rule == "self_declared_default"


# --- Rules: certification tiers (T4 / T5 / T8) -------------------------------


def test_certification_with_a_verification_marker_is_t4():
    claim = _claim(
        "AWS certified",
        SourceType.UPWORK_TEXT,
        span_text="AWS Certified Solutions Architect - proctored exam, credential ID 12345",
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T4
    assert result.tier_rule == "certification_verifiable"
    assert result.weight == pytest.approx(0.75)


def test_certification_with_a_named_issuer_but_no_verification_is_t5():
    claim = _claim(
        "completed a data analytics certificate",
        SourceType.UPWORK_TEXT,
        span_text="Completed the Google Data Analytics Certificate on Coursera",
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T5
    assert result.tier_rule == "certification_self_paced_platform"
    assert result.weight == pytest.approx(0.55)


def test_named_self_paced_platform_caps_at_t5_even_with_a_verification_marker():
    """The edge case this reordering fixes: a self-paced MOOC completion
    routinely issues a 'credential ID' as standard practice - that must NOT
    be mistaken for proctored verification and bought up to T4. Per spec
    section 3, T5 ('badge only or self-paced course') applies regardless of
    whatever else the text mentions."""
    claim = _claim(
        "completed a data analytics certificate",
        SourceType.UPWORK_TEXT,
        span_text=(
            "Completed the Google Data Analytics Certificate on Coursera, "
            "credential ID: ABC123"
        ),
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T5
    assert result.tier_rule == "certification_self_paced_platform"
    assert result.weight == pytest.approx(0.55)


def test_proctored_certification_with_no_named_self_paced_platform_still_reaches_t4():
    """Contrast case: a genuinely proctored professional exam (no MOOC
    platform named anywhere) with a verification marker must still reach
    T4 - the reordering only caps self-paced platforms, it doesn't weaken
    the T4 path for real proctored certifications."""
    claim = _claim(
        "AWS certified solutions architect",
        SourceType.UPWORK_TEXT,
        span_text="AWS Solutions Architect certified, verification ID: XYZ123",
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T4
    assert result.tier_rule == "certification_verifiable"
    assert result.weight == pytest.approx(0.75)


def test_certification_that_cannot_be_verified_drops_to_t8():
    """The explicit case this stage exists to catch: a bare, unverifiable
    claim of certification - no proctoring/verification marker, no named
    issuer - is functionally indistinguishable from any other self-
    declaration, so it drops all the way to T8, not to T5's 'badge only'."""
    claim = _claim(
        "AWS certified",
        SourceType.UPWORK_TEXT,
        span_text="I am AWS certified",
        provisional_tier=EvidenceTier.T6,  # even if ingestion guessed something stronger
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T8
    assert result.tier_rule == "certification_unverifiable"
    assert result.weight == pytest.approx(0.15)


def test_unverifiable_certification_overrides_cv_source_too():
    """The certification rules are checked BEFORE the CV-provisional-
    preserved rule, so a CV claim mentioning an unverifiable certification
    still drops to T8 rather than keeping the CV default."""
    claim = _claim(
        "AWS certified professional",
        SourceType.CV,
        span_text="AWS Certified Professional",
        provisional_tier=EvidenceTier.T6,
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T8
    assert result.tier_rule == "certification_unverifiable"


# --- Rule: cv_provisional_preserved (refines, not overwrites blindly) -------


def test_cv_history_pass_provisional_t6_is_preserved_with_no_stronger_signal():
    claim = _claim(
        "led a team of five engineers",
        SourceType.CV,
        span_text="Led a team of five engineers on a payments rewrite",
        provisional_tier=EvidenceTier.T6,
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T6
    assert result.tier_rule == "cv_provisional_preserved"
    assert result.weight == pytest.approx(0.50)


def test_cv_skills_pass_provisional_t8_is_also_preserved_not_recomputed():
    """Proves this is genuinely a REFINEMENT (deferring to what ingestion
    already knew from pass_name, which this stage can't reconstruct), not a
    coincidence of also landing on T8 via the generic self-declared default -
    the rule name must specifically be cv_provisional_preserved."""
    claim = _claim(
        "Python, SQL, Docker",
        SourceType.CV,
        span_text="Python, SQL, Docker",
        provisional_tier=EvidenceTier.T8,
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T8
    assert result.tier_rule == "cv_provisional_preserved"
    assert result.tier_rule != "self_declared_default"


def test_assign_tiers_refines_a_provisional_t2_upward_when_stronger_corroboration_exists():
    """The explicitly requested refinement test: a GitHub repo claim (T2 from
    ingestion) whose grounded span ALSO contains client-verified-outcome
    language must be REFINED (upgraded) to T1 - proving assign_tiers uses the
    provisional tier as a starting point that can be strengthened by a
    stronger signal ingestion's narrower pass-based check couldn't see, not
    just left alone or blindly recomputed to a generic default."""
    claim = _claim(
        "cut onboarding time for a client",
        SourceType.GITHUB_REPO,
        span_text="Client said this repo cut their onboarding time by 80 percent",
        provisional_tier=EvidenceTier.T2,
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T1
    assert result.tier_rule == "client_verified_outcome"
    assert result.weight == pytest.approx(1.00)


# --- Rule: peer_endorsed (T7) ------------------------------------------------


def test_peer_endorsement_language_on_linkedin_export_is_t7():
    claim = _claim(
        "recommended by a former colleague",
        SourceType.LINKEDIN_EXPORT,
        span_text="I highly recommend Jason for backend engineering roles",
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T7
    assert result.tier_rule == "peer_endorsed"
    assert result.weight == pytest.approx(0.30)


def test_linkedin_export_without_endorsement_language_is_self_declared():
    claim = _claim(
        "backend engineer",
        SourceType.LINKEDIN_EXPORT,
        span_text="Backend engineer focused on distributed systems",
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T8
    assert result.tier_rule == "self_declared_default"


# --- Rule: self_declared_default (T8) ----------------------------------------


def test_upwork_text_without_review_language_is_self_declared():
    claim = _claim(
        "experienced automation specialist",
        SourceType.UPWORK_TEXT,
        span_text="Experienced automation specialist available for hire",
    )
    result = _refined(claim)
    assert result.evidence_tier == EvidenceTier.T8
    assert result.tier_rule == "self_declared_default"


# --- Cross-cutting behavior ---------------------------------------------------


def test_assign_tiers_pulls_weight_from_tier_weights_config_for_every_rule():
    from app.config.weights import TIER_WEIGHTS

    claim = _claim("x", SourceType.GITHUB_REPO, span_text="some grounded text here")
    result = _refined(claim)
    assert result.weight == TIER_WEIGHTS[result.evidence_tier.value]


def test_assign_tiers_does_not_mutate_the_input_claim():
    original = _claim(
        "led a team", SourceType.CV, span_text="Led a team", provisional_tier=EvidenceTier.T6
    )
    assign_tiers([original])
    assert original.evidence_tier == EvidenceTier.T6
    assert original.tier_rule is None


def test_assign_tiers_preserves_claim_count_and_order():
    claims = [
        _claim("a", SourceType.GITHUB_REPO, span_text="span a text here"),
        _claim("b", SourceType.CV, span_text="span b text here", provisional_tier=EvidenceTier.T6),
        _claim("c", SourceType.UPWORK_TEXT, span_text=None),
    ]
    result = assign_tiers(claims)
    assert len(result) == 3
    assert [c.claim_text for c in result] == ["a", "b", "c"]
