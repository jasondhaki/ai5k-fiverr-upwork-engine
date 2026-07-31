"""
FastAPI app: the walking skeleton's entry point.

By the end of day one there must be a URL that takes an input and shows a
score, even if every number in it is invented. This file is that URL. It wires
the pipeline to a browser and renders the Result. Nothing here changes as real
components replace stubs - the pipeline returns the same Result shape either way.

The API contract is deliberately shaped as if generation is async (you could
return a job_id and poll), so moving to a background queue for the hosted
version does not change the interface. For local/sprint use it runs inline.

Run locally:
    uvicorn app.platform.api:app --reload
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config.weights import validate_config
from app.ingestion.extractor import validate_llm_config
from app.platform.pipeline import PipelineInput, run_pipeline
from app.storage.repository import repository

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

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

DEFAULT_NICHE = "SMB workflow automation"


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """The input form."""
    return templates.TemplateResponse(
        request, "index.html", {"niche": DEFAULT_NICHE}
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    niche: str = Form(DEFAULT_NICHE),
    github_username: str = Form(""),
    upwork_text: str = Form(""),
    cv: UploadFile | None = None,
) -> HTMLResponse:
    """
    Run the whole pipeline and render the report. Today the numbers are stubbed;
    the page is real. Replace stubs in pipeline.py and this endpoint is unchanged.
    """
    cv_bytes = await cv.read() if cv is not None else None
    inp = PipelineInput(
        niche=niche,
        cv_bytes=cv_bytes,
        github_username=github_username or None,
        upwork_text=upwork_text or None,
    )
    # Passing `repository` makes run_pipeline persist the Result AND its
    # originating Claims together, under the single run_id it mints - see
    # run_pipeline's docstring in pipeline.py. Persisted through the
    # Repository interface only (app/storage/repository.py), never raw
    # SQLAlchemy - mirrors how file_store is the only thing that ever
    # touches disk/object storage.
    result = run_pipeline(inp, repository=repository)
    response = templates.TemplateResponse(request, "result.html", {"r": result})
    # Not surfaced in the UI yet (no template change in this pass) - but
    # having *some* way to recover run_id from a real request is the whole
    # point of wiring this at all, so it doesn't only live in the DB.
    response.headers["X-Run-Id"] = result.run_id
    return response


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
