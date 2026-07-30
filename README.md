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

One deployable app, modular monolith, one database (Postgres + pgvector when you
add persistence). Storage of original files is behind an interface so local disk
swaps for S3 in one line.

## The layout

| Path | What it holds |
|---|---|
| `app/schemas/` | The three frozen record shapes. Everything imports these. |
| `app/config/weights.py` | Tier + dimension weights, efficacy table, both caps. |
| `app/storage/` | File storage interface (disk now, object store later). |
| `app/ingestion/` | Router, parsers, extractor, **span grounding**. |
| `app/evidence/` | Tier assignment rules, benchmark loading. |
| `app/scoring/` | Seven dimensions, caps, gap ranking. |
| `app/generation/` | Asset generator + span validator (build together). |
| `app/platform/` | Pipeline orchestrator, FastAPI, integration. |
| `data/benchmarks/` | Versioned hand-built benchmark JSON. |
| `tests/` | Span round-trip, schema invariants, skeleton path. |

## The rule that governs every other decision

If a claim cannot be traced to a source, it does not appear on the profile.
Everything else in this codebase exists to make that rule practical at scale.
