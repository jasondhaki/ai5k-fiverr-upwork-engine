# Build guide for Claude Code

This repo is a **working scaffold**, not an empty project. The spine already
runs end to end on stub data. Your job is to replace stubs with real components,
one at a time, left to right, without ever changing the contracts between them.

Read this file before writing code.

## Spec of record

`docs/spec.md` is the original system specification (transcribed from the
source PDF) — read it for the full rationale behind any stage, not just
what to build but why. Section numbers referenced below (and in code
comments like `config/weights.py`'s "section 3 of the spec") match its
section numbers exactly:

| Section | Covers |
|---|---|
| [1](docs/spec.md#1-system-overview) | System overview, the claim record, the governing rule |
| [2](docs/spec.md#2-data-ingestion) | Data ingestion — routing, extraction, span grounding |
| [3](docs/spec.md#3-evidence-classification) | Evidence tiers — the full T1–T8 table and rationale |
| [4](docs/spec.md#4-benchmark-engine) | Benchmark engine — anchor + radar tracks, benchmark record |
| [5](docs/spec.md#5-scoring) | Scoring — seven dimensions, keyword coverage |
| [6](docs/spec.md#6-gap-ranking) | Gap ranking — gain/priority, blocking, dependencies |
| [7](docs/spec.md#7-keeping-the-system-current) | The learning loop — refit, shrinkage, human review gate |
| [8](docs/spec.md#8-content-generation) | Content generation — generator + validator |
| [9](docs/spec.md#9-presence-engine) | Presence engine (post-slice) |

**`docs/spec.md` is the spec of record; `CLAUDE.md` is authoritative for how
the code actually works today.** The two deliberately diverge in at least
one place: spec section 2 describes the extractor returning model-computed
indices. That was implemented, tested against a live model, and found not to
work — LLMs cannot reliably compute character offsets (measured error on a
7,898-character README ranged from −2393 to +3193 with no consistent
direction). The implementation instead has the model return a verbatim quote
and locates it in the source with `str.find()` itself. `docs/spec.md` marks
this with its own `IMPLEMENTATION NOTE` at the point of divergence — the spec
text itself is left unedited so the original intent stays legible, rather
than rewritten to match what got built. When the two disagree anywhere else,
trust this file and the code, not the spec.

## The one rule that governs everything

**If a claim cannot be traced to a source span, it does not appear on the
profile.** ([spec section 1](docs/spec.md#1-system-overview)) Every
architectural decision here exists to make that rule practical. When in
doubt, re-read `app/ingestion/span_grounding.py` and its test. That test
must stay green. If you ever make it pass by weakening the assertion, you have
broken the product.

## What already works (do not rebuild)

- `app/schemas/` — the three frozen record shapes (Claim, Benchmark, Result).
  **Add optional fields if you must; never rename or remove one.**
- `app/config/weights.py` — tier weights, dimension weights, efficacy table,
  both caps. Data, not logic. Tune values here; don't scatter constants in code.
- `app/storage/store.py` — file storage behind an interface. `B2FileStore`
  (Backblaze B2, S3-compatible via boto3) is the production `FileStore`;
  `LocalFileStore` (disk) remains the dev default. `STORAGE_BACKEND=local|b2`
  picks between them - never hardcode which one, always go through the
  `file_store` singleton.
- `app/storage/db.py` + `app/storage/models.py` + `app/storage/repository.py`
  — persistence behind a `Repository` interface, the same pattern as
  `FileStore`. Postgres + pgvector on Neon (`DATABASE_URL`), falls back to a
  local SQLite file when unset. One table per frozen schema (claims,
  benchmarks, results); nested structures stay JSONB rather than fully
  normalized. Migrations live in `alembic/` - `alembic upgrade head` creates
  or updates the schema; nothing in `app/storage/` calls
  `Base.metadata.create_all()` against a real database itself.
- `app/ingestion/span_grounding.py` — the index round-trip. The core invariant.
- `app/ingestion/pdf_extractor.py` — CV text extraction behind a
  `PdfTextExtractor` protocol. `pypdfium2` (free-tier default, ~74MB peak) vs
  `docling` (layout-aware production path, ~786MB peak, needs
  `requirements-production.txt`). `PDF_PARSER=pypdfium2|docling` picks
  between them — see "Swappable backends" below.
- `app/platform/pipeline.py` — the orchestrator. Six stubbed stages.
- `app/platform/api.py` + `templates/` — the walking skeleton UI.
- `tests/` — 19 passing tests. Keep them green; add to them.

Run the skeleton:  `uvicorn app.platform.api:app --reload`
Run the tests:     `pytest -q`

## Swappable backends

Three places in this codebase deliberately keep more than one real,
working implementation behind an interface, and pick which one actually
runs from an environment variable — never a code change:

| What | Interface | Env var | Free-tier default | Production option |
|---|---|---|---|---|
| LLM provider | `LLMClient` (`app/ingestion/extractor.py`) | `LLM_PROVIDER` | either, whichever key is set | `anthropic` or `groq` |
| File storage | `FileStore` (`app/storage/store.py`) | `STORAGE_BACKEND` | `local` (disk) | `b2` (Backblaze B2) |
| PDF text extraction | `PdfTextExtractor` (`app/ingestion/pdf_extractor.py`) | `PDF_PARSER` | `pypdfium2` | `docling` (layout-aware) |

The principle: **a production-grade implementation stays in the codebase as
a real, tested code path — it is never deleted in favor of a leaner one, and
the leaner one is never a stub standing in for "build this properly later."**
Both sides are real. Going to production is an environment-variable change
on an already-working deploy, not a rewrite. This is what let the PDF parser
move from Docling-only to pypdfium2-default in response to a real hosting
memory constraint without touching the layout-aware path at all — it's still
there, still tested, one env var away.

Two things every entry in this table shares, and every new one should:

1. **Fail loudly at startup, not first use, on misconfiguration.** Each
   singleton is built once at import time (`file_store`, `pdf_text_extractor`)
   or validated once at API startup (`validate_llm_config`) — an unrecognized
   value, or a production option selected without its dependency installed,
   raises immediately with a message naming exactly what's wrong. A
   misconfigured backend must never look like a successful deploy that fails
   mysteriously on the first real request.
2. **Heavy production-only dependencies live in an optional requirements
   file** (`requirements-production.txt`), installed on top of
   `requirements.txt`, never inside it — and the code that needs them
   imports them lazily (inside a constructor or function, never at module
   top level), so the base install and the fast test suite never require
   them.

## Testing conventions

Three pytest markers gate tests out of routine iteration, registered in
`pyproject.toml`:

- `slow` — exercises real Docling PDF conversion (only the `docling`-backend
  half of the CV parser's conformance tests; the `pypdfium2`-backend half,
  and everything else in `test_cv_parser.py`, is fast and unmarked). Just
  takes time (seconds) and needs `requirements-production.txt` installed -
  see "Swappable backends" below - no external cost otherwise.
- `live_api` — makes real calls to a live external provider: LLM inference
  (Anthropic or Groq) or object storage (Backblaze B2). Costs real quota/money
  and, for the LLM providers, can trigger rate-limit backoff lasting minutes
  to **hours** on a free tier — a single `live_api` run has taken as long as
  26 minutes under backoff. Never run as part of routine iteration.
- `db` — connects to the real Postgres database (Neon, via `DATABASE_URL`)
  instead of the fast suite's in-memory SQLite. Run deliberately, e.g. after
  touching `app/storage/models.py` or an Alembic migration — never routinely,
  since it both requires network access and writes to the real hosted DB.

Day-to-day loop: `pytest -q -m "not slow and not live_api and not db"` — fast,
offline, safe to run constantly. Full suite including Docling:
`pytest -q -m "not live_api and not db"`. Everything, including live-API and
real-DB tests: plain `pytest -q`.

Run `live_api` and `db` tests **deliberately** — before a release, or after
touching extraction/grounding logic or the persistence layer specifically —
never as a matter of course. Groq's free-tier rate limits are shared across
all usage on the account, so if you've been making manual live API calls
earlier in the same session (debugging, probing quota headroom, running
`scripts/inspect_claims.py` against real sources), let that cool down before
running `live_api` tests — otherwise the tests inherit whatever quota
exhaustion you already caused and can hang for a very long time waiting out
backoff that has nothing to do with the test itself.

## Replace stubs in this exact order

Each stage lives as a stubbed function in `app/platform/pipeline.py`. Replace
the function body; keep its signature and return type. The Result shape the page
renders never changes.

1. **`extract_claims`** (ingestion, module `app/ingestion/`) — [spec section 2](docs/spec.md#2-data-ingestion)
   Build: deterministic router → per-source parser → parallel LLM extractor.
   - Router is model-free: inspect file type / text layer / size and dispatch.
   - CV parsing behind a swappable `PdfTextExtractor` backend
     (`app/ingestion/pdf_extractor.py`), picked by
     `PDF_PARSER=pypdfium2|docling` (defaults to `pypdfium2`) — see
     "Swappable backends" below and `docs/spec.md`'s `IMPLEMENTATION NOTE`
     under section 2. Support a handful of common layouts; **fail loudly**
     on anything else with a clear message. Do not chase two-column PDFs with
     photos in them.
   - GitHub via REST/GraphQL. Upwork via pasted-text parse.
   - Extractor runs several small parallel calls (identity / history / skills),
     each schema-validated and retried on failure. Every field nullable.
   - The extractor returns a **verbatim quote**, never model-computed indices —
     LLMs are reliably bad at character-offset arithmetic even over a few
     hundred characters (measured empirically: errors of dozens to thousands of
     characters, with no consistent pattern). Your own code locates that quote
     in the source via plain string search and calls `ground_span(...)` to
     re-slice the literal substring at the indices it computed. A claim whose
     quote isn't a verbatim substring drops to a coaching prompt — it is not
     published. This is the one deliberate divergence from `docs/spec.md`
     (see its `IMPLEMENTATION NOTE` under section 2, "Extraction") — the spec
     describes the original model-returned-indices design; it didn't survive
     contact with a live model, and this file's word on how it actually works
     is the one to trust.

2. **`load_benchmark`** (evidence, module `app/evidence/`) — [spec section 4](docs/spec.md#4-benchmark-engine)
   Load the versioned JSON from `data/benchmarks/`. Read a specific version,
   never "the latest". One niche is enough for the slice.

3. **`assign_tiers`** (evidence, module `app/evidence/`) — [spec section 3](docs/spec.md#3-evidence-classification)
   **Rules-based, not model-based.** A claim's tier is a deterministic function
   of its source_type and corroboration. You must be able to explain any tier.
   Pull the weight for the assigned tier from `config/weights.TIER_WEIGHTS`. The
   spec's full tier table lives at the link above, including why certifications
   split across T4/T5 and why T2 (project demonstrated) outranks T4
   (certification) - shipped work is harder to fabricate than a certificate.

4. **`score_profile`** (scoring, module `app/scoring/`) — [spec section 5](docs/spec.md#5-scoring)
   Seven pure functions `(claims, benchmark) -> dimension_score`, deterministic
   and unit-tested. Then `apply_caps` as an explicit post-step: evidence caps at
   30 and readiness caps at 30 when all claims are self-declared. Keyword
   coverage = 70% required-term presence + 30% semantic coverage (embeddings via
   pgvector or a local model); presence not frequency, hard cap on repetition.

5. **`rank_gaps`** (scoring, module `app/scoring/`) — [spec section 6](docs/spec.md#6-gap-ranking)
   Four composable steps: compute gain/priority → pull blocking items into their
   own list → apply dependency gating → assemble the balanced top five (top 3 by
   priority + largest single gain + anything blocking). Target is the benchmark
   value, not 100. efficacy comes from `config/weights.EFFICACY_TABLE`, never
   computed live.

6. **`generate_assets`** (generation, module `app/generation/`) — [spec section 8](docs/spec.md#8-content-generation)
   Build the generator and the span validator **together, in the same change**.
   The validator re-reads each source span (via `reverify_span`) and confirms
   every number in the generated text traces to a span. Structure it so a
   generated asset cannot become part of a Result without passing the validator —
   make skipping it structurally impossible, not a matter of remembering to call
   it. Set `validated=True` only after it passes.

## After the slice works

Widen ingestion (LinkedIn, portfolio, video, articles) → automate the benchmark
(anchor monthly + radar daily, radar produces tags only, never weights) → add
persistence (Postgres + pgvector, move off in-memory) → add the async job queue
for hosting → build the learning loop LAST (offline batch, ridge with
non-negative coefficients, shrink toward prior, human review gate, never an
automatic silent rollout). Corresponds to [spec section 7](docs/spec.md#7-keeping-the-system-current)
(the learning loop), [section 9](docs/spec.md#9-presence-engine) (presence
engine, ships after this), and [section 11](docs/spec.md#11-build-order)
(the stage-by-stage build order this whole plan is derived from).

## Conventions

- One deployable app, modular monolith. Do not split into microservices.
- Everything crossing a layer boundary is a validated Pydantic model instance.
- Anthropic SDK directly for LLM calls. No LangChain — it abstracts the exact
  part (the span round-trip) you most need to control.
- Empty output in the right shape is fine. Wrong shape is not.
- Write the test with the component, not after it.
