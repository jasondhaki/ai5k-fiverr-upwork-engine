"""
The router: model-free dispatch from PipelineInput to per-source parsers.

Every decision here is made by inspecting which fields are populated (and, for
the CV, its file type / text layer) - never by asking a model. A source that's
absent is skipped; a source that's present but malformed fails loudly instead
of silently contributing zero claims, so a bad upload never looks like "the
freelancer just has no evidence."
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from app.ingestion.cv_parser import parse_cv
from app.ingestion.github_parser import parse_github
from app.ingestion.upwork_parser import parse_upwork_text
from app.schemas import Claim

if TYPE_CHECKING:
    from app.platform.pipeline import PipelineInput


def route_input(
    inp: "PipelineInput", on_status: Callable[..., None] | None = None
) -> list[Claim]:
    """
    Dispatch each populated source in `inp` to its parser and concatenate
    the resulting claims. Order: CV, GitHub, Upwork - matches intake form.

    `on_status`, if given, is passed straight through to whichever parsers
    actually run (each reports its own stage/detail), and this function
    reports the running claims_found/claims_provable total after each
    source - the only place that CAN, since no single parser knows about
    the others' contributions.
    """
    claims: list[Claim] = []

    if inp.cv_bytes:
        claims.extend(parse_cv(inp.cv_bytes, on_status=on_status))
        if on_status is not None:
            on_status(
                claims_found=len(claims),
                claims_provable=sum(1 for c in claims if c.publishable),
            )

    if inp.github_username:
        claims.extend(parse_github(inp.github_username, on_status=on_status))
        if on_status is not None:
            on_status(
                claims_found=len(claims),
                claims_provable=sum(1 for c in claims if c.publishable),
            )

    if inp.upwork_text:
        claims.extend(parse_upwork_text(inp.upwork_text, on_status=on_status))
        if on_status is not None:
            on_status(
                claims_found=len(claims),
                claims_provable=sum(1 for c in claims if c.publishable),
            )

    return claims
