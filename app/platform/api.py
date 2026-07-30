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

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config.weights import validate_config
from app.platform.pipeline import PipelineInput, run_pipeline

validate_config()  # fail loudly at startup if weights are inconsistent

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
    result = run_pipeline(inp)
    return templates.TemplateResponse(request, "result.html", {"r": result})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
