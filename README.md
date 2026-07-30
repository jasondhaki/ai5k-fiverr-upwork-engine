# AI5K Profile Intelligence — Working Slice

A freelancer uploads a CV, gives a GitHub username, and pastes their existing
profile text. They get back a readiness score, a ranked list of gaps, and a
rewritten title and overview **where every number can be traced to where it came
from.** That last clause is the whole product.

This repo is a running scaffold: the full pipeline path works end to end on stub
data today. Real components replace stubs one at a time. See `CLAUDE.md` for the
build order.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # 19 tests should pass
uvicorn app.platform.api:app --reload       # open http://127.0.0.1:8000
```

Fill the form and submit. You'll get a full report — capped readiness score,
seven dimensions, blocking items shown separately, top gaps ranked by points per
hour, and a rewritten title. Every number is currently invented; the shape is
real. That is the walking skeleton, and it's the point.

## Architecture at a glance

```
        input (CV, GitHub, Upwork paste)
                    |
      [router] deterministic, model-free           app/ingestion/
                    |
      [parsers] one per source type
                    |
      [extractor] parallel LLM calls, returns INDICES
                    |
      [span_grounding] re-slices literal substring  <-- the core invariant
                    |
                 Claim[]  ----------------------------> app/schemas/claim.py
                    |
      [assign_tiers] rules-based, explainable        app/evidence/
      [load_benchmark] versioned, immutable
                    |
      [score_profile] 7 pure dimension fns + caps    app/scoring/
      [rank_gaps] gain/priority + blocking + deps
                    |
      [generate_assets] generator + span VALIDATOR   app/generation/
                    |
                 Result  -----------------------------> app/schemas/result.py
                    |
      [FastAPI + templates] the report page          app/platform/
```

One deployable app, modular monolith, one database (Postgres + pgvector on
Neon). Storage of original files is behind an interface (`app/storage/store.py`)
so local disk swaps for Backblaze B2 with one env var; persistence of claims,
benchmarks, and results is behind a Repository interface
(`app/storage/repository.py`) so nothing outside `app/storage/` ever issues a
raw SQLAlchemy call.

## The layout

| Path | What it holds |
|---|---|
| `app/schemas/` | The three frozen record shapes. Everything imports these. |
| `app/config/weights.py` | Tier + dimension weights, efficacy table, both caps. |
| `app/storage/` | File storage (disk / B2) + DB persistence (Repository), both behind interfaces. |
| `app/ingestion/` | Router, parsers, extractor, **span grounding**. |
| `app/evidence/` | Tier assignment rules, benchmark loading. |
| `app/scoring/` | Seven dimensions, caps, gap ranking. |
| `app/generation/` | Asset generator + span validator (build together). |
| `app/platform/` | Pipeline orchestrator, FastAPI, integration. |
| `alembic/` | DB migrations - `alembic upgrade head` creates/updates the schema. |
| `data/benchmarks/` | Versioned hand-built benchmark JSON. |
| `tests/` | Span round-trip, schema invariants, skeleton path. |

## Environment variables

Set these in `.env` for local dev (never committed - see `.gitignore`), or in
Render's dashboard for the hosted deploy (see "Deploying to Render" below).

| Variable | Purpose | Required when |
|---|---|---|
| `GITHUB_TOKEN` | GitHub REST/GraphQL auth for the GitHub parser | Always (ingestion) |
| `LLM_PROVIDER` | `anthropic` or `groq` - picks the extractor/generator's LLM client | Always (extraction/generation) |
| `ANTHROPIC_API_KEY` | Anthropic API key | `LLM_PROVIDER=anthropic` |
| `GROQ_API_KEY` | Groq API key | `LLM_PROVIDER=groq` |
| `DATABASE_URL` | Postgres connection string (Neon) | Persistence; falls back to a local SQLite file if unset |
| `STORAGE_BACKEND` | `local` or `b2` - picks the FileStore backend | Defaults to `local`; set `b2` for the hosted deploy |
| `B2_KEY_ID` | Backblaze B2 application key ID | `STORAGE_BACKEND=b2` |
| `B2_APPLICATION_KEY` | Backblaze B2 application key | `STORAGE_BACKEND=b2` |
| `B2_ENDPOINT` | B2 S3-compatible endpoint, bare hostname (e.g. `s3.us-east-005.backblazeb2.com`) - the `https://` scheme is prepended in code, not stored here | `STORAGE_BACKEND=b2` |
| `B2_BUCKET_NAME` | B2 bucket name | `STORAGE_BACKEND=b2` |

## Deploying to Render

The service deploys as a Docker image (`render.yaml` + `Dockerfile`), not
Render's native Python runtime - Docling's `[standard]` extra pulls in torch,
onnxruntime, and opencv, which need system libraries (`libgl1`,
`libglib2.0-0`, `libgomp1`) the native runtime's image doesn't ship and gives
no way to install. See the Dockerfile's comments for exactly why each package
is there.

One-time setup before the first deploy:

1. Create a Neon Postgres project and copy its connection string.
2. Create a Backblaze B2 bucket and an application key scoped to it.
3. In the Render dashboard, create the Blueprint from `render.yaml`, then set
   every `sync: false` env var it lists (`DATABASE_URL`, `B2_*`,
   `LLM_PROVIDER`, `ANTHROPIC_API_KEY`/`GROQ_API_KEY`, `GITHUB_TOKEN`) -
   these are deliberately not committed anywhere in this repo.

Every deploy runs `alembic upgrade head` against `DATABASE_URL` first (see
`render.yaml`'s `preDeployCommand`), so schema changes ship with the code that
needs them. `STORAGE_BACKEND=b2` is set in `render.yaml` itself (not a
secret) so the hosted service uses Backblaze B2 rather than local disk, which
wouldn't survive a redeploy anyway.

## The rule that governs every other decision

If a claim cannot be traced to a source, it does not appear on the profile.
Everything else in this codebase exists to make that rule practical at scale.
