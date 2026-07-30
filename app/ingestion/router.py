"""
The router: model-free dispatch from PipelineInput to per-source parsers.

Every decision here is made by inspecting which fields are populated (and, for
the CV, its file type / text layer) - never by asking a model. A source that's
absent is skipped; a source that's present but malformed fails loudly instead
of silently contributing zero claims, so a bad upload never looks like "the
freelancer just has no evidence."
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.ingestion.cv_parser import parse_cv
from app.ingestion.github_parser import parse_github
from app.ingestion.upwork_parser import parse_upwork_text
from app.schemas import Claim

if TYPE_CHECKING:
    from app.platform.pipeline import PipelineInput


def route_input(inp: "PipelineInput") -> list[Claim]:
    """Dispatch each populated source in `inp` to its parser and concatenate
    the resulting claims. Order: CV, GitHub, Upwork - matches intake form."""
    claims: list[Claim] = []

    if inp.cv_bytes:
        claims.extend(parse_cv(inp.cv_bytes))

    if inp.github_username:
        claims.extend(parse_github(inp.github_username))

    if inp.upwork_text:
        claims.extend(parse_upwork_text(inp.upwork_text))

    return claims
