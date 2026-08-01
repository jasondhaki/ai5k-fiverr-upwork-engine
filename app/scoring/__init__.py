"""
Scoring: seven dimension functions + the evidence cap (M3, spec section 5),
and gap ranking (M3, spec section 6).

    from app.scoring import score_profile, rank_gaps
"""

from app.scoring.caps import apply_caps
from app.scoring.dimensions import completeness_checklist_status, keyword_term_status
from app.scoring.gaps import rank_gaps
from app.scoring.profile import score_profile
from app.scoring.skill_gaps import find_skill_gaps

__all__ = [
    "score_profile",
    "apply_caps",
    "rank_gaps",
    "find_skill_gaps",
    "keyword_term_status",
    "completeness_checklist_status",
]
