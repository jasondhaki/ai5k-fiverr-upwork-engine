"""
Tier assignment: the real, rules-based refinement of ingestion's provisional
tiers (spec section 3).

Ingestion (app/ingestion/extractor.py::_provisional_tier) already assigns a
PROVISIONAL tier per claim from a narrow, source-type/pass-based lookup - the
best guess available with only one claim in view, no cross-cutting text
analysis, and no visibility into corroboration signals that live in the
claim's own grounded text (a quoted client outcome, a verifiable
certification, a peer endorsement). This stage sees the full, already-
deduplicated, already-boilerplate-filtered claims list and applies the fuller
rules table - it does not re-solve either of those upstream problems.

Every rule below is named, and every claim leaving this stage carries which
rule decided its tier (`Claim.tier_rule`), plus a log line - "explainable"
here means "point at the specific rule that fired", not "plausible in the
aggregate."

No model calls anywhere: a tier is decided by whether a source_span exists at
all, by source_type, or by a plain regex match against the claim's own
GROUNDED span text (never an ungrounded paraphrase) - see `_rules_tier` for
the exact cascade, checked strongest tier first.

REFINE, don't blindly re-derive: most rules below fire only on a signal
ingestion's narrower pass couldn't see (client-outcome language regardless of
source, certification verifiability, peer endorsement, platform assessment).
Where no such signal exists, the previous stage's tier is generally
reproduced anyway (e.g. a plain GitHub repo claim recomputes to T2 either
way) - except for CV claims specifically, where ingestion's provisional tier
already encodes a distinction (work-history vs. a bare skill list, via which
extraction PASS produced the claim) that this stage cannot reconstruct, since
pass_name isn't preserved on the Claim object itself. For CV claims with no
stronger override signal, the provisional tier is kept as-is rather than
recomputed from source_type alone, which would silently collapse that
distinction.
"""

from __future__ import annotations

import logging
import re

from app.config.weights import TIER_WEIGHTS
from app.schemas import Claim, EvidenceTier, SourceType

logger = logging.getLogger(__name__)

# A quoted client outcome - a review, a testimonial, a star rating, the
# client themselves named as the source of a result. Checked against the
# GROUNDED span text (plus the model's claim_text, as a secondary signal -
# see _rules_tier), for ANY source type, not just Upwork: ingestion's own
# check only looks at Upwork text, so this is the fuller version that lets a
# GitHub repo or CV claim with the same corroborating language upgrade too.
_CLIENT_OUTCOME_PATTERN = re.compile(
    r"\b(review|testimonial|rated|star rating|\d\s*-\s*star|"
    r"client (said|wrote|noted|reported))\b",
    re.IGNORECASE,
)

# Source types where the "evidence" IS the demonstrated artifact itself - a
# deployed repo, a live app, a published model - per EvidenceTier.T2's own
# docstring ("Project demonstrated: deployed repo, live app, published model").
_PROJECT_DEMONSTRATED_SOURCES = frozenset({
    SourceType.GITHUB_REPO,
    SourceType.PORTFOLIO_SITE,
    SourceType.HUGGINGFACE,
    SourceType.DEMO_VIDEO,
})

# Our own structured skills assessment, not a bare self-description - only
# promotes an ONBOARDING_FORM claim when the text specifically reads like an
# assessment RESULT, so plain intake-form prose still falls through to
# self-declared.
_PLATFORM_ASSESSMENT_PATTERN = re.compile(
    r"\b(skills? assessment|assessment score|platform assessment|"
    r"verified skill level)\b",
    re.IGNORECASE,
)

# Certification language, refined into three outcomes below by how checkable
# it actually is - and, checked FIRST, by whether it's a self-paced/non-
# proctored platform completion at all:
#   - names a specific, known self-paced platform (Coursera, Udemy, edX,
#     etc.) -> T5, ALWAYS, even when a verification marker (e.g. a
#     "credential ID") is also present. Self-paced course platforms issue a
#     credential ID as standard practice on completion - that's routine
#     record-keeping, not proof of proctoring, so it must not let a
#     credential-ID-shaped string alone buy its way up to T4. Per spec
#     section 3: T5 is "badge only or self-paced course", full stop,
#     regardless of what else the text happens to mention.
#   - otherwise (no named self-paced platform), names an explicit
#     verification mechanism (proctored exam, a credential ID, a
#     verification hub) -> T4, the strongest tier a certification can
#     reach - this is the AWS/PMP-style proctored professional exam case,
#     not a MOOC completion.
#   - mentions "certified"/"certification" with NEITHER -> T8. A bare,
#     unverifiable claim of certification is functionally indistinguishable
#     from any other self-declaration, so it does not get to keep whatever
#     tier ingestion's source-type default happened to assign.
_CERT_MENTION_PATTERN = re.compile(
    r"\b(certificat(?:e|ion)|certified|credential|badge)\b", re.IGNORECASE
)
_CERT_VERIFICATION_MARKER_PATTERN = re.compile(
    r"\b(proctored|verify(?:\s+(?:at|this|it))?|verification\s+(?:id|code|link)|"
    r"credential\s?id|credly\.com|badge\s?id)\b",
    re.IGNORECASE,
)
# Deliberately NOT cloud-provider brand names like "AWS" or "Azure": naming
# WHICH certification you claim to have ("I am AWS certified") says nothing
# about whether it's self-paced or proctored - it still names nothing a
# reviewer could actually go look up. Naming the completion PLATFORM ("...on
# Coursera") is what identifies it as self-paced and caps it at T5 - checked
# BEFORE the verification-marker check, not merely as its fallback.
_CERT_NAMED_ISSUER_PATTERN = re.compile(
    r"\b(coursera|udemy|linkedin\s+learning|pluralsight|edx|skillshare|"
    r"hackerrank|codecademy)\b",
    re.IGNORECASE,
)

# A LinkedIn recommendation/endorsement is peer-sourced by definition - only
# promotes when the text actually reads as one, so an exported LinkedIn
# summary/about section (the person's own words) still falls through to
# self-declared rather than getting credit for a peer's endorsement it isn't.
_PEER_ENDORSEMENT_PATTERN = re.compile(
    r"\b(recommend(?:s|ed|ation)?|endorsed|endorsement)\b", re.IGNORECASE
)


def _rules_tier(claim: Claim) -> tuple[EvidenceTier, str]:
    """
    THE fuller rules table (spec section 3), evaluated as a strict priority
    cascade - strongest tier checked first, first rule that matches wins.
    Returns (tier, rule_name) so the caller can stamp which rule fired.

    Order matters and encodes real priority, not just convenience: project-
    demonstrated sources are checked BEFORE any certification pattern, so a
    GitHub repo claim that also happens to mention a certification still
    lands on T2, never falls to T4/T5/T8 over it - shipped work outranks a
    certification mention, per config/weights.py's own comment on T2.
    """
    if not claim.publishable:
        return EvidenceTier.T8, "not_grounded"

    # claim.source_span is guaranteed non-None past this point (publishable
    # implies source_span is not None). Checked against both the grounded
    # span text (the only text that's actually proven) and the model's own
    # claim_text (a secondary signal only - never used to publish anything,
    # only to help classify evidence that's already grounded).
    haystack = f"{claim.claim_text}\n{claim.source_span.text}"

    if _CLIENT_OUTCOME_PATTERN.search(haystack):
        return EvidenceTier.T1, "client_verified_outcome"

    if claim.source_type in _PROJECT_DEMONSTRATED_SOURCES:
        return EvidenceTier.T2, "project_demonstrated"

    if claim.source_type == SourceType.ONBOARDING_FORM and _PLATFORM_ASSESSMENT_PATTERN.search(
        haystack
    ):
        return EvidenceTier.T3, "platform_assessed"

    if _CERT_MENTION_PATTERN.search(haystack):
        # Named self-paced platform checked FIRST and caps at T5 even if a
        # verification marker (e.g. a credential ID) is also present - a
        # MOOC completion certificate routinely has one as standard
        # practice, and that is not the same claim as a proctored exam.
        if _CERT_NAMED_ISSUER_PATTERN.search(haystack):
            return EvidenceTier.T5, "certification_self_paced_platform"
        if _CERT_VERIFICATION_MARKER_PATTERN.search(haystack):
            return EvidenceTier.T4, "certification_verifiable"
        return EvidenceTier.T8, "certification_unverifiable"

    if claim.source_type == SourceType.CV:
        # Ingestion's provisional tier already distinguishes work-history
        # (T6) from a bare skill list (T8) via which pass produced the
        # claim - information not preserved on the Claim object itself, so
        # it can't be reconstructed here. No stronger override fired above,
        # so keep it rather than silently recomputing a worse answer.
        return claim.evidence_tier, "cv_provisional_preserved"

    if claim.source_type == SourceType.LINKEDIN_EXPORT and _PEER_ENDORSEMENT_PATTERN.search(
        haystack
    ):
        return EvidenceTier.T7, "peer_endorsed"

    return EvidenceTier.T8, "self_declared_default"


def assign_tiers(claims: list[Claim]) -> list[Claim]:
    """
    Refine every claim's provisional evidence_tier using the fuller rules
    table above. Claims arriving here are already deduplicated and
    boilerplate-filtered by ingestion (M1) - this stage only refines tiers,
    it never re-solves either of those. Returns a new list; input claims are
    never mutated.
    """
    refined: list[Claim] = []
    for claim in claims:
        tier, rule_name = _rules_tier(claim)
        logger.info(
            "claim %r: %s -> %s via rule %r",
            claim.claim_text,
            claim.evidence_tier.value,
            tier.value,
            rule_name,
        )
        refined.append(
            claim.model_copy(
                update={
                    "evidence_tier": tier,
                    "weight": TIER_WEIGHTS[tier.value],
                    "tier_rule": rule_name,
                }
            )
        )
    return refined
