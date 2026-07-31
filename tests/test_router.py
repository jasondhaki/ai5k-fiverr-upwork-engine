"""
Router tests: dispatch is model-free and driven purely by which PipelineInput
fields are populated. Every parser is monkeypatched so this test never
touches Docling, the network, or an LLM.
"""

from __future__ import annotations

import app.ingestion.router as router
from app.platform.pipeline import PipelineInput
from app.schemas import Claim, SourceSpan, SourceType
from app.schemas.claim import EvidenceTier


def _fake_claim(tag: str) -> Claim:
    return Claim(
        claim_text=tag,
        source_type=SourceType.UPWORK_TEXT,
        source_span=SourceSpan(document_id="d", start_index=0, end_index=len(tag), text=tag),
        evidence_tier=EvidenceTier.T8,
        weight=0.15,
    )


def test_router_calls_nothing_when_no_sources_present(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(router, "parse_cv", lambda b, **kw: calls.append("cv") or [])
    monkeypatch.setattr(router, "parse_github", lambda u, **kw: calls.append("github") or [])
    monkeypatch.setattr(router, "parse_upwork_text", lambda t, **kw: calls.append("upwork") or [])

    claims = router.route_input(PipelineInput(niche="n"))

    assert calls == []
    assert claims == []


def test_router_dispatches_only_present_sources(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        router, "parse_cv", lambda b, **kw: calls.append("cv") or [_fake_claim("cv")]
    )
    monkeypatch.setattr(router, "parse_github", lambda u, **kw: calls.append("github") or [])
    monkeypatch.setattr(
        router,
        "parse_upwork_text",
        lambda t, **kw: calls.append("upwork") or [_fake_claim("upwork")],
    )

    inp = PipelineInput(niche="n", cv_bytes=b"%PDF-fake", upwork_text="some pasted text")
    claims = router.route_input(inp)

    assert calls == ["cv", "upwork"]
    assert [c.claim_text for c in claims] == ["cv", "upwork"]


def test_router_dispatches_github_when_username_present(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(router, "parse_cv", lambda b, **kw: calls.append("cv") or [])
    monkeypatch.setattr(
        router, "parse_github", lambda u, **kw: calls.append("github") or [_fake_claim("github")]
    )
    monkeypatch.setattr(router, "parse_upwork_text", lambda t, **kw: calls.append("upwork") or [])

    inp = PipelineInput(niche="n", github_username="octocat")
    claims = router.route_input(inp)

    assert calls == ["github"]
    assert claims[0].claim_text == "github"


def test_router_reports_running_claims_totals_after_each_source(monkeypatch):
    """route_input is the only place that sees the FULL picture across
    sources, so it - not any individual parser - must report the running
    claims_found/claims_provable total, once per source, as extraction
    proceeds."""
    monkeypatch.setattr(router, "parse_cv", lambda b, **kw: [_fake_claim("cv")])
    monkeypatch.setattr(router, "parse_upwork_text", lambda t, **kw: [_fake_claim("upwork")])

    updates: list[dict] = []
    inp = PipelineInput(niche="n", cv_bytes=b"%PDF-fake", upwork_text="some pasted text")
    router.route_input(inp, on_status=lambda **fields: updates.append(fields))

    claims_updates = [u for u in updates if "claims_found" in u]
    assert [u["claims_found"] for u in claims_updates] == [1, 2]
    assert [u["claims_provable"] for u in claims_updates] == [1, 2]
