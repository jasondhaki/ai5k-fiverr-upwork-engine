"""
The seven scoring dimensions (spec section 5), as pure functions:

    (claims: list[Claim], benchmark: Benchmark) -> float   # 0-100

Each function is deterministic and side-effect free - no I/O, no model calls,
no clock reads (recency is already baked into Claim.effective_weight by the
time claims reach here). Every function is independently unit-testable with
hand-built Claim/Benchmark fixtures.

IMPORTANT CAVEAT, read before trusting these numbers: the current schema has
no representation of "the profile" as separately-authored text (a title, an
overview paragraph, a stated hourly rate, a filled-in checklist). It only has
`claims` (individual pieces of evidence) and `benchmark` (what good looks
like). FOUR of the seven dimensions - positioning, completeness, conversion,
and pricing strategy - are properly about that authored text/those profile
fields, which do not exist yet in this data model (it's what stage 6,
`generate_assets`, will eventually produce). Each of the four is implemented
below as an honest, clearly-labeled HEURISTIC over the evidence available
now, not a placeholder that fakes a number - each one's own docstring below
has a PROVISIONAL block spelling out exactly what it approximates, what it
will need once a real profile object exists, and that its score should be
treated as provisional until then. Search this file for "PROVISIONAL" to
find all four at once.
"""

from __future__ import annotations

import re

from app.config.weights import KEYWORD_REQUIRED_WEIGHT, KEYWORD_SEMANTIC_WEIGHT
from app.schemas import Benchmark, Claim, SourceType

# --- Shared helpers -----------------------------------------------------------

_NUMBER_PATTERN = re.compile(r"\d")

_STOPWORDS = frozenset({
    "with", "that", "this", "from", "were", "have", "been", "into", "over",
    "using", "used", "your", "their", "which", "about", "these", "those",
    "will", "would", "could", "should", "there", "where", "while", "and",
})


def _significant_words(text: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{3,}", text)
        if word.lower() not in _STOPWORDS
    }


def _claim_text_pool(claims: list[Claim]) -> str:
    """All text available for vocabulary-style matching: the model's own
    claim_text plus, where grounded, the literal proven span text. This is
    the closest available stand-in for "the profile's vocabulary" until a
    real authored-profile object exists (see module docstring)."""
    parts: list[str] = []
    for claim in claims:
        parts.append(claim.claim_text)
        if claim.source_span is not None:
            parts.append(claim.source_span.text)
    return "\n".join(parts)


def _contains(term: str, text_lower: str) -> bool:
    """Presence, not frequency - a term matched once counts exactly the same
    as one matched nine times, per spec section 5's hard cap on repetition.
    Plain case-folded substring match (not regex) so terms with punctuation
    like "Make.com" don't need escaping."""
    return term.lower() in text_lower


# --- 1. Positioning (22%) -----------------------------------------------------

_ROLE_TITLE_PATTERN = re.compile(
    r"\b(engineer|developer|specialist|consultant|designer|architect|"
    r"analyst|manager|freelancer)\b",
    re.IGNORECASE,
)


def score_positioning(claims: list[Claim], benchmark: Benchmark) -> float:
    """
    Spec: "Role, vertical and outcome stated with enough specificity to be
    found and believed." Approximated here as three independent evidence-
    derived signals, each worth a third of the score:
      1. a role/title word appears somewhere in the evidence
      2. the evidence vocabulary overlaps this niche's own benchmark terms
         (i.e. there IS evidence relevant to the vertical being scored for)
      3. at least one claim states a concrete, numeric outcome

    PROVISIONAL - HEURISTIC, NOT A MEASUREMENT. This is not measuring the
    profile's positioning at all - it's measuring whether role/vertical/
    outcome words show up ANYWHERE across the raw evidence, which says
    nothing about whether the profile's actual title and opening line state
    them "with enough specificity to be found and believed" (they could
    appear once, buried, in claim #9 of 11, and this would still score it
    100). It NEEDS the real authored title and overview text once stage 6
    (`generate_assets`) produces one, and should be rewritten to check that
    text directly. Treat this score as PROVISIONAL until then.
    """
    if not claims:
        return 0.0

    pool = _claim_text_pool(claims)
    pool_lower = pool.lower()

    has_role = bool(_ROLE_TITLE_PATTERN.search(pool))

    niche_terms = [*benchmark.required_terms, *benchmark.benchmark_topics]
    has_vertical_alignment = any(_contains(term, pool_lower) for term in niche_terms)

    has_outcome = bool(_NUMBER_PATTERN.search(pool))

    signals_met = sum([has_role, has_vertical_alignment, has_outcome])
    return (signals_met / 3) * 100.0


# --- 2. Evidence quality (22%) ------------------------------------------------


def score_evidence_quality(claims: list[Claim], benchmark: Benchmark) -> float:
    """
    Spec section 5: "Sum of tier weights across claimed skills, subject to
    the cap." (The cap itself is applied separately, in apply_caps - this
    function returns the RAW, uncapped score.) A skill takes its strongest
    evidence, never the sum of weak evidence (EvidenceTier's own rule): for
    each distinct skill_id, take the best effective_weight among claims that
    are actually PUBLISHABLE (unproven claims contribute no evidence quality
    no matter what tier ingestion guessed). Claims with no skill_ids don't
    contribute - this dimension is explicitly about claimed SKILLS.

    DELIBERATE DIVERGENCE FROM THE SPEC'S LITERAL WORDING: the spec line
    above says "sum", but this function computes the AVERAGE of per-skill
    best weights, not a running total. Reasoning: every dimension function
    in this file has a 0-100 contract, and a literal, un-normalized sum
    grows without bound as more skills are claimed - five skills each near
    weight 1.0 would sum to ~5.0, meaning it would ALREADY need some
    normalization step (e.g. dividing by skill count, or by some assumed
    max skill count) before it could be a 0-100 score at all. "Sum, then
    normalize by dividing by the number of skills" and "average" are the
    same arithmetic operation - this function just performs that
    normalization directly as the mean, rather than computing an
    intermediate unbounded sum and dividing it by count as a separate step.
    No information is lost by doing it this way (both give the same 0-100
    number), but the divergence from the spec's literal word "sum" is real
    and deliberate, so it's spelled out explicitly here rather than left
    for a future reader to notice as an unexplained discrepancy.
    """
    best_by_skill: dict[str, float] = {}
    for claim in claims:
        if not claim.publishable:
            continue
        for skill_id in claim.skill_ids:
            best_by_skill[skill_id] = max(
                best_by_skill.get(skill_id, 0.0), claim.effective_weight
            )

    if not best_by_skill:
        return 0.0

    average = sum(best_by_skill.values()) / len(best_by_skill)
    return min(100.0, average * 100.0)


# --- 3. Keyword coverage (15%) ------------------------------------------------

# PLACEHOLDER for the real embedding-based semantic match (spec section 5:
# "Matched by meaning, not by string... Pinecone counts as vector database").
# This is CPU-only local dev with no hosted embeddings API - a small,
# explicit synonym map plus a significant-word-overlap fallback stands in for
# now. Replace with a real embedding similarity (pgvector or a local
# sentence-transformers model) before this number is trusted for real users;
# it will systematically under-count coverage phrased in unlisted synonyms.
_TOPIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "vector database": ("pinecone", "weaviate", "qdrant", "milvus", "chroma", "vector db", "vectordb"),
    "no-code integration": ("zapier", "make.com", "n8n", "workato"),
    "workflow automation": ("automate", "automation", "automated workflow"),
    "crm automation": ("salesforce", "hubspot", "pipedrive"),
    "data pipelines": ("etl", "elt", "data pipeline"),
    "error handling and retries": ("retry", "retries", "exponential backoff", "error handling"),
}


def _topic_covered(topic: str, text_lower: str) -> bool:
    if _contains(topic, text_lower):
        return True
    for synonym in _TOPIC_SYNONYMS.get(topic.lower(), ()):
        if _contains(synonym, text_lower):
            return True
    # Fallback: any significant word from the topic phrase shows up at all.
    # Crude, but a floor under the explicit synonym map above.
    return any(word in text_lower for word in _significant_words(topic))


def score_keyword_coverage(claims: list[Claim], benchmark: Benchmark) -> float:
    """
    70% required-term presence (exact match) + 30% semantic coverage of
    benchmark_topics (see _topic_covered's placeholder note). Both halves
    are presence-only, never frequency - a term matched once scores
    identically to one matched nine times, so there is no way to raise this
    score by repeating a keyword (spec section 5's explicit hard cap).
    """
    pool_lower = _claim_text_pool(claims).lower()

    if benchmark.required_terms:
        present = sum(1 for term in benchmark.required_terms if _contains(term, pool_lower))
        required_score = present / len(benchmark.required_terms)
    else:
        required_score = 1.0

    if benchmark.benchmark_topics:
        covered = sum(1 for topic in benchmark.benchmark_topics if _topic_covered(topic, pool_lower))
        semantic_score = covered / len(benchmark.benchmark_topics)
    else:
        semantic_score = 1.0

    return (KEYWORD_REQUIRED_WEIGHT * required_score + KEYWORD_SEMANTIC_WEIGHT * semantic_score) * 100.0


def keyword_term_status(claims: list[Claim], benchmark: Benchmark) -> list[dict[str, object]]:
    """
    Per-required-term presence, for display (the report page lists which
    required terms are present vs missing explicitly, not just a rolled-up
    score) - the exact same _contains check score_keyword_coverage's
    required-term half already runs internally, just surfaced per-term
    instead of collapsed into one aggregate number. Order matches
    benchmark.required_terms.
    """
    pool_lower = _claim_text_pool(claims).lower()
    return [
        {"term": term, "present": _contains(term, pool_lower)}
        for term in benchmark.required_terms
    ]


# --- 4. Portfolio quality (15%) -----------------------------------------------

# Source types where the evidence IS a portfolio item - a deployed repo, a
# live site, a published model, a demo. Same taxonomy fact as
# app/evidence/tiers.py's project-demonstrated set (T2), duplicated locally
# rather than imported: this module scores a DIFFERENT thing (portfolio
# presentation quality) from a coincidentally-related fact (tier), and
# importing a private constant across module boundaries for that would be
# the wrong kind of coupling.
_PORTFOLIO_SOURCES = frozenset({
    SourceType.GITHUB_REPO,
    SourceType.PORTFOLIO_SITE,
    SourceType.HUGGINGFACE,
    SourceType.DEMO_VIDEO,
})


def score_portfolio_quality(claims: list[Claim], benchmark: Benchmark) -> float:
    """
    Spec: "Item count, quantified results, working links." Item count and
    quantified results are directly derivable from claims. "Working links"
    would need a live HTTP check, which a pure, I/O-free function can't do -
    approximated instead by requiring the claim be PUBLISHABLE (grounded):
    ingestion already had to successfully reach and parse that source to
    produce a grounded claim from it at all, which is the closest available
    stand-in for "the link worked" without doing a network call here.

    Item/quantified counts are keyed by source_span.document_id, not by claim
    - deliberately, not incidentally. Different extraction passes routinely
    ground two genuinely different claims (e.g. a skill-used claim and a
    project-demonstrated claim) to the exact same source sentence; counting
    per-claim would double-count that single piece of evidence as two
    portfolio items. Keying by document_id means every claim grounded
    anywhere in the same source document - whether they share the identical
    span or not - collapses to one item, matching "one repo/one document is
    one portfolio item" regardless of how many claims cite it. See
    test_portfolio_quality_counts_two_claims_sharing_an_identical_span_once.
    """
    portfolio_claims = [
        c for c in claims if c.source_type in _PORTFOLIO_SOURCES and c.publishable
    ]
    if not portfolio_claims:
        return 0.0

    item_ids = {c.source_span.document_id for c in portfolio_claims}
    quantified_ids = {
        c.source_span.document_id
        for c in portfolio_claims
        if _NUMBER_PATTERN.search(c.claim_text) or _NUMBER_PATTERN.search(c.source_span.text)
    }

    item_score = (
        min(1.0, len(item_ids) / benchmark.portfolio_min_items)
        if benchmark.portfolio_min_items
        else 1.0
    )
    quantified_score = (
        min(1.0, len(quantified_ids) / benchmark.portfolio_min_quantified)
        if benchmark.portfolio_min_quantified
        else 1.0
    )
    return ((item_score + quantified_score) / 2) * 100.0


# --- 5. Completeness (10%) -----------------------------------------------------


def score_completeness(claims: list[Claim], benchmark: Benchmark) -> float:
    """
    Spec: "Checklist of essential profile elements." Checks four evidence-
    derived proxies for the elements that checklist would cover: identity/
    history evidence, demonstrated-project evidence, a minimum breadth of
    distinct skills, and client-facing self-description text. Score is
    simply how many of the four are present, out of four.

    PROVISIONAL - HEURISTIC, NOT A MEASUREMENT. This is not checking a
    profile-field checklist at all - there IS no such checklist yet (no
    title field, no overview field, no rate field, no portfolio-link field
    to check "filled in or not"). It substitutes "has ingestion seen this
    KIND of evidence" for "has this profile FIELD been filled in", which are
    different claims - a person could have rich CV evidence and still have
    left their actual profile title blank. It NEEDS the real profile record
    (title/overview/rate/portfolio fields) once stage 6 (`generate_assets`)
    produces one, and should be rewritten to check those fields directly.
    Treat this score as PROVISIONAL until then.
    """
    has_identity_history = any(
        c.source_type in (SourceType.CV, SourceType.LINKEDIN_EXPORT) for c in claims
    )
    has_portfolio_evidence = any(c.source_type in _PORTFOLIO_SOURCES for c in claims)
    distinct_skills = {skill for c in claims for skill in c.skill_ids}
    has_skill_breadth = len(distinct_skills) >= 3
    has_client_facing_text = any(
        c.source_type in (SourceType.UPWORK_TEXT, SourceType.ONBOARDING_FORM) for c in claims
    )

    checklist = [has_identity_history, has_portfolio_evidence, has_skill_breadth, has_client_facing_text]
    return (sum(checklist) / len(checklist)) * 100.0


# --- 6. Conversion (8%) --------------------------------------------------------

_OUTCOME_FRAMING_PATTERN = re.compile(
    r"\b(cut|saved|reduced|increased|improved|grew|automated|eliminated|"
    r"delivered|shipped)\b",
    re.IGNORECASE,
)


def score_conversion(claims: list[Claim], benchmark: Benchmark) -> float:
    """
    Spec: "Whether the opening addresses the buyer's problem rather than
    describing the seller." Approximated as the share of claims that are
    framed around a concrete outcome/action (result-oriented language or a
    number) rather than a bare self-description, as a rough proxy for the
    person's general tendency to talk in outcomes versus in self-description.

    PROVISIONAL - HEURISTIC, NOT A MEASUREMENT. This is not reading "the
    opening" at all - there is no opening text to read yet. It's averaging
    outcome-language across ALL claims and treating that average as a stand-
    in for how one specific sentence (the profile's actual opening line)
    will read. A profile could open with "I am a hard working developer"
    while every OTHER claim underneath is outcome-framed, and this would
    still score well - the two things aren't the same measurement. It NEEDS
    the real authored overview text once stage 6 (`generate_assets`)
    produces one, and should be rewritten to check its opening sentence(s)
    directly. Treat this score as PROVISIONAL until then.
    """
    if not claims:
        return 0.0
    outcome_framed = sum(
        1
        for c in claims
        if _OUTCOME_FRAMING_PATTERN.search(c.claim_text) or _NUMBER_PATTERN.search(c.claim_text)
    )
    return (outcome_framed / len(claims)) * 100.0


# --- 7. Pricing strategy (8%) --------------------------------------------------


def score_pricing_strategy(claims: list[Claim], benchmark: Benchmark) -> float:
    """
    Spec: "Whether the stated rate is defensible given the evidence tier."
    Approximated as the share of publishable claims whose evidence_tier is
    one of the tiers the benchmark's own rate_band lists as justifying it
    (`benchmark.rate_band.justifying_tiers`, e.g. T1/T2). This directly
    reuses real benchmark data rather than inventing a separate threshold,
    even without a stated rate to check it against yet.

    PROVISIONAL - HEURISTIC, NOT A MEASUREMENT. This is not checking whether
    a stated rate is defensible at all - there is no stated rate anywhere in
    this data model to check. It substitutes "what share of the evidence
    would justify the top rate band" for the real question, "does the
    person's ACTUAL stated number fall inside the band their evidence
    justifies" - a person could score 100 here while stating a rate wildly
    above or below what their evidence supports, and this function would
    never know. It NEEDS the real stated hourly rate once stage 6
    (`generate_assets`) produces a profile that carries one, and should be
    rewritten to compare that rate against `benchmark.rate_band` directly.
    Treat this score as PROVISIONAL until then.
    """
    grounded = [c for c in claims if c.publishable]
    if not grounded:
        return 0.0
    justifying_tiers = set(benchmark.rate_band.justifying_tiers)
    matching = sum(1 for c in grounded if c.evidence_tier.value in justifying_tiers)
    return (matching / len(grounded)) * 100.0
