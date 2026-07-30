"""
Scoring: seven dimension functions + the evidence cap (M3, spec section 5),
and gap ranking (M3, spec section 6).

    from app.scoring import score_profile, rank_gaps
"""

from app.scoring.caps import apply_caps
from app.scoring.gaps import rank_gaps
from app.scoring.profile import score_profile

__all__ = [
    "score_profile",
    "apply_caps",
    "rank_gaps",
]
