"""
FastAPI app: the walking skeleton's entry point.

By the end of day one there must be a URL that takes an input and shows a
score, even if every number in it is invented. This file is that URL. It wires
the pipeline to a browser and renders the Result. Nothing here changes as real
components replace stubs - the pipeline returns the same Result shape either way.

/analyze is asynchronous: POST kicks the pipeline off as an in-process
FastAPI BackgroundTask and redirects immediately to a progress page that
polls /analyze/{run_id}/status - a real pipeline run makes several live LLM
calls and can take anywhere from a few seconds to (on Groq's free tier,
rate-limited) over a minute, and holding the request open for that with zero
feedback both looks broken and risks the connection dying from a platform
timeout even though the work would have succeeded. See app/platform/status.py
for the status-tracking store, and pipeline.py's run_pipeline for the
on_status callback both write to.

Deliberately FastAPI's built-in BackgroundTasks, not Celery/Redis/an external
queue - this is a free-tier demo; an in-process background task is the right
weight. KNOWN LIMITATION: if this process restarts mid-run, the background
task and its status both simply stop existing - there is no queue durability
here. See status.py's docstring for why that's an acceptable, coherent
failure mode rather than something needing a fix in this pass.

Run locally:
    uvicorn app.platform.api:app --reload
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config.weights import validate_config
from app.evidence import load_benchmark
from app.ingestion.extractor import validate_llm_config
from app.platform.pipeline import PipelineInput, run_pipeline
from app.platform.status import status_store
from app.scoring import completeness_checklist_status, keyword_term_status
from app.storage.repository import repository

logger = logging.getLogger(__name__)

validate_config()  # fail loudly at startup if weights are inconsistent
# Fail loudly at startup if LLM_PROVIDER/its matching API key is missing too -
# otherwise the container starts and /health passes even with no usable LLM
# client, and the first real analyze request fails with a generic SDK error
# instead of a clear one. See validate_llm_config's docstring. Note this does
# mean importing this module (as tests/test_api.py's TestClient does) needs
# SOME LLM credential present in the environment - any non-empty value works,
# no real/live key required, since this never makes a network call.
validate_llm_config()

app = FastAPI(title="AI5K Profile Intelligence - Working Slice")

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(REPO_ROOT / "static")), name="static")

DEFAULT_NICHE = "SMB workflow automation"


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """The input form."""
    return templates.TemplateResponse(
        request, "index.html", {"niche": DEFAULT_NICHE}
    )


def _run_pipeline_in_background(inp: PipelineInput, run_id: str) -> None:
    """
    The actual pipeline run, executed by FastAPI's BackgroundTasks after the
    redirect response has already gone out - there is no HTTP response left
    to raise an error INTO by the time something here fails, so this is the
    one place that must catch everything and turn it into a status update
    instead: an unhandled exception here would just vanish into the
    background-task machinery with nothing to show the person watching the
    progress page except an eternal spinner.

    `run_pipeline` itself keeps raising on bad input for every direct caller
    (tests, scripts) exactly as before - this wrapper is where "running this
    as a background job" specifically becomes this endpoint's concern.
    """

    def on_status(**fields: object) -> None:
        status_store.update(run_id, **fields)

    try:
        run_pipeline(inp, repository=repository, on_status=on_status, run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - genuinely must catch everything here
        logger.exception("Pipeline run %s failed", run_id)
        status_store.update(run_id, stage="error", error=str(exc))


@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    niche: str = Form(DEFAULT_NICHE),
    github_username: str = Form(""),
    upwork_text: str = Form(""),
    cv: UploadFile | None = None,
) -> RedirectResponse:
    """
    Mint run_id, register it in the status store, kick the pipeline off in
    the background, and redirect immediately to its progress page - see this
    module's docstring for why. 303 (not the default 307) so the browser
    issues a GET against the redirect target instead of re-POSTing the form.
    """
    cv_bytes = await cv.read() if cv is not None else None
    inp = PipelineInput(
        niche=niche,
        cv_bytes=cv_bytes,
        github_username=github_username or None,
        upwork_text=upwork_text or None,
    )
    run_id = uuid.uuid4().hex
    status_store.create(run_id)
    background_tasks.add_task(_run_pipeline_in_background, inp, run_id)
    return RedirectResponse(url=f"/analyze/{run_id}", status_code=303)


@app.get("/analyze/{run_id}", response_class=HTMLResponse)
def analyze_progress(request: Request, run_id: str) -> HTMLResponse:
    """The progress page - polls /analyze/{run_id}/status and redirects to
    the report itself once stage == 'done' (see templates/progress.html)."""
    if status_store.get(run_id) is None:
        raise HTTPException(status_code=404, detail=f"No run found for run_id {run_id!r}")
    return templates.TemplateResponse(request, "progress.html", {"run_id": run_id})


@app.get("/analyze/{run_id}/status")
def analyze_status(run_id: str) -> dict:
    """What the progress page polls. 404 for a run_id nobody ever created -
    distinct from a run that's merely still queued/running, which is a
    normal 200 with stage='queued' or whatever it's currently doing.

    Replaces the stored absolute `rate_limit_resume_at` with a freshly
    computed `rate_limit_seconds_remaining` on every call, against this
    server's own clock - the browser's clock is never asked to agree with
    it, so a slow poll or a backgrounded tab never produces a stale or
    negative countdown on the progress page.
    """
    status = status_store.get(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"No run found for run_id {run_id!r}")
    body = status.model_dump(mode="json")
    resume_at = body.pop("rate_limit_resume_at")
    body["rate_limit_seconds_remaining"] = (
        max(0.0, resume_at - time.time()) if resume_at is not None else None
    )
    return body


@app.get("/analyze/{run_id}/result", response_class=HTMLResponse)
def analyze_result(request: Request, run_id: str) -> HTMLResponse:
    """
    The same report page /analyze used to render inline before the async
    rewrite - now reached only once a background run has actually finished
    and persisted, via the Repository (there is no other copy of the Result
    lying around once the background task that computed it has returned).

    Re-loads the exact benchmark this Result was scored against (by the
    (niche, benchmark_version) it stamped at generation time - never "the
    latest") purely for display: the full per-dimension targets, the
    required-terms present/missing breakdown, and the rate band, none of
    which Result itself needs to carry since the benchmark is a versioned,
    load-any-time artifact rather than per-run data.

    Also fetches the underlying claims - not just to compute keyword_status,
    but so the "fix before publishing" section can show exactly which claims
    are unproven, not just a count, reusing the same claim-card markup as
    the /claims audit trail (see templates/_claim_card.html).
    """
    result = repository.get_result(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No result found for run_id {run_id!r}")
    claims = repository.get_claims(run_id)
    benchmark = load_benchmark(result.niche, result.benchmark_version)
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "r": result,
            "benchmark": benchmark,
            "keyword_status": keyword_term_status(claims, benchmark),
            "completeness_status": completeness_checklist_status(claims, benchmark),
            "unproven_claims": [c for c in claims if not c.publishable],
        },
    )


@app.get("/analyze/{run_id}/claims", response_class=HTMLResponse)
def analyze_claims(request: Request, run_id: str) -> HTMLResponse:
    """
    The claim-level audit trail behind one report: every claim, its evidence
    tier, its source, and the literal span text it grounds against - grouped
    by source the same way scripts/inspect_claims.py already formats for
    manual review, just rendered for a browser instead of a terminal. This is
    the system's core promise made checkable: not a summary number, but the
    actual provenance, reusing data the Repository already persists rather
    than any new backend logic.
    """
    report = repository.get_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"No report found for run_id {run_id!r}")
    result, claims = report
    grouped: dict[str, list] = {}
    for claim in claims:
        grouped.setdefault(claim.source_type.value, []).append(claim)
    return templates.TemplateResponse(
        request, "claims.html", {"r": result, "grouped": grouped}
    )


@app.get("/report/{run_id}")
def get_report(run_id: str) -> dict:
    """
    Retrieve a persisted report by its run_id: the scored Result and the
    exact Claims that produced it, together. JSON only, no UI yet - the
    foundation for a future "show me the source" feature, and for auditing
    what a real run actually produced after the fact.
    """
    report = repository.get_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"No report found for run_id {run_id!r}")
    result, claims = report
    return {
        "result": result.model_dump(mode="json"),
        "claims": [claim.model_dump(mode="json") for claim in claims],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
