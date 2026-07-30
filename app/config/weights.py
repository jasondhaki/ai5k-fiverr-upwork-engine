"""
Versioned configuration: tier weights, dimension weights, and the efficacy
table. These are DATA, not logic. They are kept out of the scoring functions so
they can be tuned - and eventually refit from outcomes - without touching code.

The efficacy table in particular is "a guess wearing the clothes of arithmetic"
if you re-estimate it per run. It is a fixed, versioned lookup, improved
deliberately over time as outcome data accumulates, never computed live.

CONFIG_VERSION stamps every score. When the learning loop proposes new weights,
it writes a NEW version behind a human review gate - it never mutates these.
"""

from datetime import date

CONFIG_VERSION = "2026-07-hand-set"

# --- Evidence tier weights (section 3 of the spec) -------------------------
# A skill takes its strongest evidence; these are the per-tier multipliers.
TIER_WEIGHTS: dict[str, float] = {
    "T1": 1.00,  # Client-verified outcome
    "T2": 0.85,  # Project demonstrated (shipped work outranks certification)
    "T3": 0.80,  # Platform-assessed
    "T4": 0.75,  # Certification, proctored
    "T5": 0.55,  # Certification, badge only
    "T6": 0.50,  # Employer-confirmed
    "T7": 0.30,  # Peer-endorsed
    "T8": 0.15,  # Self-declared
}

# --- Dimension weights (section 5 of the spec) -----------------------------
# Must sum to 1.0. Positioning + evidence together are 44%, deliberately.
DIMENSION_WEIGHTS: dict[str, float] = {
    "positioning": 0.22,
    "evidence_quality": 0.22,
    "keyword_coverage": 0.15,
    "portfolio_quality": 0.15,
    "completeness": 0.10,
    "conversion": 0.08,
    "pricing_strategy": 0.08,
}

# --- The two caps (section 3, "the evidence cap") --------------------------
# When every claim is self-declared, evidence caps here and readiness caps too.
EVIDENCE_DIMENSION_CAP = 30.0  # out of 100 (i.e. 3 out of 10)
READINESS_CAP_WHEN_UNPROVEN = 30.0  # overall ceiling with no real proof

# --- Keyword coverage split (section 5) ------------------------------------
KEYWORD_REQUIRED_WEIGHT = 0.70  # exact required-term presence
KEYWORD_SEMANTIC_WEIGHT = 0.30  # semantic coverage of benchmark topics

# --- Efficacy table (section 6) --------------------------------------------
# Expected share of a gap that a given fix type actually closes. Keyed by
# dimension. Hand-set for the sprint; refit deliberately later. NEVER live.
EFFICACY_TABLE: dict[str, float] = {
    "positioning": 0.80,
    "evidence_quality": 0.60,
    "keyword_coverage": 0.90,
    "portfolio_quality": 0.70,
    "completeness": 0.95,
    "conversion": 0.75,
    "pricing_strategy": 0.85,
}

# --- Gap ranking constants (section 6) -------------------------------------
EFFORT_HOURS_FLOOR = 0.5  # trivial fixes cannot dominate the ranking

# Estimated hours to close a gap on this dimension, keyed by dimension. Hand-
# set for the sprint, same status as EFFICACY_TABLE: a versioned lookup,
# refit deliberately later from real completion times, never computed live.
# effort_hours in the gain/priority formula is floored at EFFORT_HOURS_FLOOR
# regardless of what's set here, so a value below the floor has no effect.
EFFORT_HOURS_TABLE: dict[str, float] = {
    "positioning": 1.5,  # rewrite title/vertical framing
    "evidence_quality": 4.0,  # go get and document real proof - the expensive one
    "keyword_coverage": 1.0,  # revise wording to cover required terms/topics
    "portfolio_quality": 3.0,  # add or write up portfolio items
    "completeness": 0.5,  # fill in missing checklist fields
    "conversion": 1.0,  # rewrite the opening around the buyer's problem
    "pricing_strategy": 0.5,  # adjust the stated rate
}

# --- Recency decay (section 3, discounting older evidence) ------------------
# Evidence halves in value every RECENCY_HALF_LIFE_DAYS, floored so old-but-real
# evidence never decays to zero - a 2019 shipped project still counts for
# something, just materially less than one shipped last month.
RECENCY_HALF_LIFE_DAYS = 365.0  # one year to halve
RECENCY_FLOOR = 0.20  # asymptote: the oldest evidence still counts this much


def recency_factor(observed_date: date | None, as_of: date | None = None) -> float:
    """
    Exponential decay from 1.0 toward RECENCY_FLOOR, halving every
    RECENCY_HALF_LIFE_DAYS of age. `observed_date` is when the evidence was
    created (e.g. a repo's last commit date); `as_of` defaults to today.

    Returns 1.0 when `observed_date` is None (nothing to discount without a
    date) or not in the past (clock skew / same-day evidence).
    """
    if observed_date is None:
        return 1.0
    today = as_of if as_of is not None else date.today()
    age_days = (today - observed_date).days
    if age_days <= 0:
        return 1.0
    decayed = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    return RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * decayed


def validate_config() -> None:
    """Fail loudly at startup if the config is internally inconsistent."""
    total = sum(DIMENSION_WEIGHTS.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Dimension weights must sum to 1.0, got {total}")
    if set(DIMENSION_WEIGHTS) != set(EFFICACY_TABLE):
        raise ValueError("Efficacy table and dimension weights must cover the same dimensions")
    if set(DIMENSION_WEIGHTS) != set(EFFORT_HOURS_TABLE):
        raise ValueError("Effort hours table and dimension weights must cover the same dimensions")
