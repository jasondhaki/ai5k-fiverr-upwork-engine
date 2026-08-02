# System Architecture

Technical reference for how the codebase is actually built, module by
module. For the plain-language "what and why" version, see
[system-overview-for-boss.md](system-overview-for-boss.md). For the original
specification and rationale, see [spec.md](spec.md). **This file describes
the code as it exists today** — where it diverges from `spec.md`, this file
and `CLAUDE.md` are authoritative (see `CLAUDE.md`'s "Spec of record"
section).

---

## 1. Shape of the system

One deployable app, a modular monolith — not microservices. FastAPI serves a
server-rendered UI (Jinja2 templates) over a six-stage pipeline. Every stage
is a pure(ish) function; the only side effects are LLM calls, file storage,
and the database, each hidden behind a small protocol.

```
PipelineInput
     │
     ▼
┌─────────────────┐   ┌──────────────┐   ┌───────────────┐
│ 1. extract_claims│──▶│ 2. load_     │──▶│ 3. assign_    │
│  (router → parsers│   │   benchmark  │   │    tiers      │
│   → LLM extractor)│   │ (versioned   │   │ (rules-based  │
│                   │   │  JSON file)  │   │  refinement)  │
└─────────────────┘   └──────────────┘   └───────┬───────┘
                                                    │
        ┌───────────────────────────────────────────┘
        ▼
┌─────────────────┐   ┌──────────────┐   ┌───────────────┐
│ 4. score_profile │──▶│ 5. rank_gaps │──▶│ 6. generate_  │
│ (7 dimensions +  │   │ (gain/priority│   │    assets     │
│  caps)           │   │  + blocking) │   │ (generator +  │
│                   │   │              │   │  validator)   │
└─────────────────┘   └──────────────┘   └───────┬───────┘
                                                    │
                                                    ▼
                                                 Result
```

`app/platform/pipeline.py::run_pipeline` is the spine that calls all six in
order and assembles the final `Result`. Every stage's function signature is
frozen — replacing a stub with a real implementation never changes the
contract on either side of it (see `CLAUDE.md`'s "Replace stubs in this
exact order").

## 2. Directory map

```
app/
  schemas/        Claim, Benchmark, Result — the three frozen record shapes
  config/         weights.py (tunable data), env.py (.env loader)
  ingestion/      router, per-source parsers, LLM extractor, span grounding,
                  PDF text extraction
  evidence/       benchmark loading, tier assignment rules
  scoring/        seven dimension functions, caps, gap ranking, skill gaps
  generation/     asset generator + span validator
  storage/        FileStore (originals/text), Repository (Postgres), ORM models
  platform/       pipeline orchestrator, FastAPI app, in-process status store
templates/        Jinja2 templates for the walking-skeleton UI
data/benchmarks/  versioned benchmark JSON files (one per niche+version)
alembic/          DB migrations
tests/            ~250 tests, gated by `slow` / `live_api` / `db` markers
```

## 3. The frozen contracts (`app/schemas/`)

Three Pydantic models cross every layer boundary. They are additive-only —
fields may be added, never renamed or removed (`CLAUDE.md`, `claim.py`'s own
docstring).

### `Claim` (`app/schemas/claim.py`)

The atomic unit of evidence. Key fields:

- `claim_text`, `skill_ids`, `source_type`
- `source_span: Optional[SourceSpan]` — origin pointer; `None` means
  unpublishable
- `evidence_tier: EvidenceTier` (T1–T8) and `weight` (from
  `config/weights.TIER_WEIGHTS`)
- `observed_date` / `recency_factor` — age discount
- `tier_rule` — which named rule in `assign_tiers` decided the tier

Two derived properties carry the whole product's governing invariant:

```python
publishable      = source_span is not None
effective_weight = weight * recency_factor
```

`SourceSpan` holds `document_id`, `start_index`, `end_index`, and `text` (the
literal re-sliced substring — never model-paraphrased), with a validator
that rejects any span whose `text` length doesn't match its index range.

### `Benchmark` (`app/schemas/benchmark.py`)

Versioned and immutable (`model_config = {"frozen": True}`). Identified by
`(niche, version)`. Carries `required_terms`, `benchmark_topics`,
`title_formula`, portfolio/overview targets, a `RateBand`, and
`dimension_targets` (per-dimension score the top tier actually reaches — not
100).

### `Result` (`app/schemas/result.py`)

Everything one pipeline run produces: capped `readiness`, seven
`DimensionScore`s, `blocking` items (kept separate from ranked `gaps`),
`generated` assets, and provenance counts (`total_claims`,
`provable_claims`). Two diagnosability flags — `generation_incomplete` and
`overview_blocked_by_evidence_tier` — distinguish "nothing to generate" from
"generation was attempted and failed," which the UI needs but the score
itself never depends on. `skill_gaps` is attached but is explicitly never a
scoring input (see §7).

## 4. Ingestion (`app/ingestion/`) — spec §2

### Router (`router.py`)

Model-free dispatch. Inspects which fields are populated on `PipelineInput`
(`cv_bytes`, `github_username`, `upwork_text`) and calls the matching parser.
A populated-but-malformed source fails loudly rather than silently
contributing zero claims.

### Per-source parsers

- **`cv_parser.py`** — PDF text extraction via the swappable
  `PdfTextExtractor` protocol (see §4a), then the shared LLM extractor.
- **`github_parser.py`** — REST/GraphQL pulls (repos, languages, activity,
  last-commit dates for recency).
- **`upwork_parser.py`** — parses pasted profile text (no scraping; the user
  supplies their own text).

### The LLM extractor (`extractor.py`)

Up to three independent passes per document — **identity**, **history**,
**skills** — run in parallel via a `ThreadPoolExecutor`. Which passes run is
chosen per `source_type` (`_PASSES_BY_SOURCE`): a CV gets all three; a
GitHub README skips "identity" (no role language in a project doc); pasted
Upwork text skips "history" (identity/skills already cover it). This is a
deliberate cost cut — unconditionally running all three on a 20-repo GitHub
profile would mean 60 LLM calls.

**The core mechanism — quote, don't compute:**

```
model returns: an "evidence_quote" (verbatim substring), never indices
     │
     ▼
_try_ground(): str.find() locates the quote in the source text
     │
     ▼
ground_span(): re-slices the literal substring at the found indices
     │
     ▼
SourceSpan.text == the source itself, never anything the model produced
```

This is `CLAUDE.md`'s documented divergence from `spec.md` §2: the spec
describes model-computed indices; empirical testing found LLMs unreliable at
character-offset arithmetic (errors from dozens to thousands of characters
on real documents), so the model only ever supplies a quote and the code
does the arithmetic. If the quote isn't an exact substring, `str.find`
returns -1 and the claim drops to `source_span=None` (unpublishable) rather
than being published against a guessed location.

Hardening on top of exact-substring matching:

- **Quote/dash normalization** (`_QUOTE_AND_DASH_NORMALIZATION`) — folds
  typographically-equivalent glyphs (curly vs. straight quotes, en/em dash
  vs. hyphen) to one canonical form each, one-code-point-to-one-code-point,
  so index positions stay valid in the original text. Fixed ~40% of observed
  ungrounded claims in testing.
- **Vocabulary-overlap check** (`_significant_words`) — a grounded quote
  must share at least one significant word with the claim text itself, so a
  model can't attach a real-but-unrelated quote to a fabricated claim just
  to satisfy the schema.
- **Boilerplate filtering** (`_is_boilerplate`) — regex/phrase-based
  detection of generic scaffolding text (`create-next-app` READMEs, bare
  `npm install`), applied to both `claim_text` and the grounded span.
- **Deduplication** (`_dedupe_claims`) — collapses exact and near-duplicate
  claims (difflib similarity ≥ 0.6) that multiple passes independently
  rediscover, with a deterministic tie-break (grounded beats ungrounded,
  then higher `effective_weight`, then longer span).

**Provisional tiering at extraction time** (`_provisional_tier`) is a narrow,
source-type/pass-based lookup — the best guess with only one claim in view.
It is later refined by `assign_tiers` (§5), which sees the full claim set and
has a fuller rules table.

**LLM provider abstraction:** `LLMClient` protocol (`complete(system,
prompt) -> str`), satisfied by `AnthropicClient` (Anthropic SDK directly, no
LangChain) and `GroqClient` (Groq's OpenAI-compatible REST endpoint, plain
`requests` — no second SDK). `build_default_client()` picks one via
`LLM_PROVIDER=anthropic|groq`, or falls back to whichever API key is set
(Anthropic first). `validate_llm_config()` fails loudly at FastAPI startup if
neither is configured, mirroring `store.py`'s B2 startup check.

`GroqClient` has its own rate-limit backoff (up to 8 attempts, honoring
`Retry-After`, capped at 60s) since free-tier Groq TPM limits are easy to
hit running three passes concurrently. Backoff state is surfaced live to the
progress page via a contextvar-propagated `on_status` callback
(`status_reporting()`), not a parameter threaded through every function —
see §8.

### PDF extraction (`pdf_extractor.py`)

Swappable backend behind `PdfTextExtractor` (`extract_text(bytes) -> str`):

| Backend | Selected by | Peak memory | Notes |
|---|---|---|---|
| `PypdfiumExtractor` | `PDF_PARSER=pypdfium2` (default) | ~74MB | Plain text-layer extraction, no layout/OCR |
| `DoclingExtractor` | `PDF_PARSER=docling` | ~786MB | Layout-aware; lazy import, needs `requirements-production.txt` |

Picked once at import time; an unrecognized value or `docling` selected
without the dependency installed fails immediately, not on first upload.

### Span grounding (`span_grounding.py`)

The single most important module in the system (its own docstring says so).
Two functions:

- `ground_span(...)` — turns indices into a verified `SourceSpan` by
  re-slicing `source_text[start:end]`. Raises `SpanGroundingError` on
  out-of-bounds or inverted indices.
- `reverify_span(span, current_source_text)` — re-checks that a *stored*
  span still resolves to the same substring against the *current* source.
  This is what the generation-time validator calls to catch drift.

Has its own test asserting the index round-trip is exact — "that test must
stay green" (`CLAUDE.md`).

## 5. Evidence (`app/evidence/`) — spec §§3–4

### Benchmark loading (`benchmarks.py`)

`load_benchmark(niche, version)` reads
`data/benchmarks/{slugified_niche}_{version}.json`. No "latest" fallback —
an exact version is always requested, and both a missing file and a
file/label mismatch raise `BenchmarkNotFoundError`. Today there is exactly
one hand-written benchmark file (`smb_workflow_automation_2026-07.json`); the
anchor-track automation described in spec §4 is not yet built (see §11).

### Tier assignment (`tiers.py`)

Rules-based, not model-based — `assign_tiers(claims)` re-derives each
claim's tier from a strict priority cascade (`_rules_tier`), strongest tier
checked first, evaluated only against the claim's own **grounded** span
text (plus `claim_text` as a secondary signal):

1. Client-outcome language (review/testimonial/rating) anywhere → **T1**
2. `source_type` is a demonstrated-project source (GitHub repo, portfolio
   site, HuggingFace, demo video) → **T2**
3. Onboarding form + platform-assessment language → **T3**
4. Certification language, refined by verifiability:
   - names a known self-paced platform (Coursera, Udemy, ...) → **T5**,
     always, even with a credential-ID-shaped string present
   - otherwise has a real verification marker (proctored, credential ID) →
     **T4**
   - otherwise, bare "certified" mention → **T8**
5. CV claims with no stronger override keep ingestion's provisional tier
   (work-history vs. bare-skill distinction can't be reconstructed here —
   `pass_name` isn't preserved on `Claim`)
6. LinkedIn export + peer-endorsement language → **T7**
7. Default → **T8**

Every claim leaving this stage carries `tier_rule` — the exact rule name
that fired — and a log line, so any tier is traceable, never "plausible in
aggregate."

## 6. Scoring (`app/scoring/`) — spec §5

### Seven dimension functions (`dimensions.py`)

Each is `(claims, benchmark) -> float` (0–100), pure, deterministic, no I/O:

| Dimension | Weight | Status |
|---|---|---|
| `score_positioning` | 22% | **PROVISIONAL** — heuristic |
| `score_evidence_quality` | 22% | Real measurement |
| `score_keyword_coverage` | 15% | Real (embeddings still placeholder-synonym-map) |
| `score_portfolio_quality` | 15% | Real measurement |
| `score_completeness` | 10% | **PROVISIONAL** — heuristic |
| `score_conversion` | 8% | **PROVISIONAL** — heuristic |
| `score_pricing_strategy` | 8% | **PROVISIONAL** — heuristic |

**Why four are marked PROVISIONAL:** the current schema has no representation
of "the authored profile" — no title field, no overview field, no stated
rate. Those four dimensions are properly about that text, which doesn't
exist until stage 6 (`generate_assets`) produces one. Each computes an
honest, labeled heuristic over raw evidence instead (e.g. `score_positioning`
checks whether role/vertical/outcome words appear *anywhere* in the
evidence pool, not whether the actual title states them). `is_provisional()`
reads the marker string directly out of each function's docstring — a single
source of truth, not a second hand-kept list that could drift.

Notable implementation details:

- `score_evidence_quality` **averages** per-skill best `effective_weight`
  rather than summing (spec says "sum") — documented as a deliberate
  divergence to keep the function on a bounded 0–100 scale.
- `score_keyword_coverage` is 70% required-term presence + 30% semantic
  coverage (`KEYWORD_REQUIRED_WEIGHT` / `KEYWORD_SEMANTIC_WEIGHT`), both
  presence-only (a term matched once == matched nine times — the spec's hard
  cap on repetition). Semantic coverage today is a small explicit synonym
  map (`_TOPIC_SYNONYMS`) plus vocabulary-overlap fallback — a placeholder
  for real embedding similarity (pgvector `Vector` column already exists on
  `BenchmarkRecord`, unpopulated — see `storage/models.py`).
- `score_portfolio_quality` keys item/quantified counts by
  `source_span.document_id`, not by claim — two claims grounded to the same
  document collapse to one portfolio item.

### Caps (`caps.py`) — spec §3

Explicit post-processing step, never folded into a dimension function:

```python
all_claims_self_declared(claims)  # every claim is T8, or there are none
        │  if True
        ▼
evidence_quality dimension capped at 30/100
overall readiness capped at 30/100
```

`cap_reason()` distinguishes "no claims at all" from "claims exist but are
all T8" — a claim can be a verbatim, spot-checkable quote and *still* be T8
(T8 means no third-party corroboration, not "ungrounded").

### Gap ranking (`gaps.py`) — spec §6

Four composable steps in `rank_gaps`:

1. **Score every candidate gap:**
   `gain = weight * (target - current) * efficacy`,
   `priority = gain / max(effort_hours, EFFORT_HOURS_FLOOR)`.
   `target` is the benchmark's per-dimension value, never 100. `efficacy`
   and `effort_hours` come from fixed lookup tables
   (`config/weights.EFFICACY_TABLE` / `EFFORT_HOURS_TABLE`), never computed
   live.
2. **Blocking items** are pulled into their own list (`BlockingItem`), never
   part of the ranking. Currently only `UNPROVEN_CLAIM` is ever raised — the
   other two `BlockingReason` values (`TOS_RISK`,
   `MISSING_IDENTITY_VERIFICATION`) exist in the enum for when a signal
   exists to raise them, but an empty blocking list today is *not* evidence
   those risks are absent (see the module's own "NOT YET IMPLEMENTED" note).
3. **Dependency gating** (`_DEPENDENCY_GATES`) — `positioning` is hidden
   until at least one claim's text overlaps the niche's own vocabulary
   (not just "any claim exists"); `pricing_strategy` is hidden until at
   least one claim is grounded at all.
4. **Assemble the balanced result** — top 3 candidates by `priority`, plus
   the single largest-`gain` candidate if not already included. Never padded
   to a fixed count; a zero-gain or gated dimension never appears.

### Skill gaps (`skill_gaps.py`) — spec §5

Deliberately **not** a scoring dimension — never imported by `score_profile`
or `rank_gaps`, never a factor in `readiness`. Reports every benchmark
`required_term`/`benchmark_topic` with **no** matching claim anywhere (via
`skill_ids` or the same `_contains`/`_topic_covered` checks
`score_keyword_coverage` uses, so it can't disagree with that dimension
about what counts as present). A pure "what to learn next" signal, computed
alongside scoring and attached to the same `Result` for convenience only.

## 7. Generation (`app/generation/`) — spec §8

Built and tested together, in the same change, per `CLAUDE.md`: the
generator and validator are two halves of one structural guarantee.

### Generator (`generator.py`)

`generate_title_draft` / `generate_overview_draft` produce a `DraftAsset` —
a type that **has no `validated` field at all** and is never itself
convertible to a `GeneratedAsset`. Evidence shown to the model is
pre-filtered by *our* code, not left to the model to self-select:

- Title: best-available claims with a numeric outcome in their grounded
  span (`_select_title_claims`), strongest first.
- Overview: only T1–T4 claims (`_select_overview_proof_claims`,
  `PROOF_TIERS = {T1, T2, T3, T4}`) — spec §8's "proof section drawn only
  from T1–T4."

Both return `None` (not an error) when there's no qualifying evidence —
"empty output in the right shape," never an invented number. The title
prompt's proof-language instruction (`tier_verified`) tells the model
whether words like "proven"/"verified" are appropriate, based on whether
*all* selected claims are T1–T4 — but this is only the first line of
defense; see the validator for the structural enforcement.

Two of spec §8's four asset kinds — `case_study` and `proposal_draft` — are
explicitly **not implemented** yet (documented, not silently missing).

### Validator (`validator.py`)

The **only** place in the codebase that constructs a `GeneratedAsset` with
`validated=True`. `validate_asset(draft) -> GeneratedAsset` runs four checks
in order, raising `AssetValidationError` naming exactly what failed:

1. **(overview only)** every `claim_ref` is T1–T4 — an independent second
   guard on top of the generator's own filtering.
2. If not every `claim_ref` is T1–T4, the draft text must not use
   proof-implying language (`proven`/`verified`/`guaranteed`) — independent
   of the generator's own prompt instruction, so a model that ignores the
   instruction is still caught.
3. Every `claim_ref`'s span **re-verifies against the current stored
   source** (`reverify_span`, re-read via `file_store.get_text`) — catches
   drift if the underlying document changed since extraction.
4. Every number in the draft text is backed by at least one `claim_ref`'s
   current span text, with real word boundaries (`"20"` must not match
   inside `"2019"`) and matching unit context (a number presented as a
   percentage must be backed by a percentage in the *same* claim's span,
   not a bare digit-substring from an unrelated context).

Deliberately biased toward over-rejection — a generic "24/7" with no real
backing gets rejected too, and that's the correct tradeoff.

## 8. Storage (`app/storage/`)

Two independent abstractions, same pattern (protocol + env-var-selected
singleton + fail-loud-at-startup):

### `FileStore` (`store.py`)

Holds original files and extracted text. Every parsed source produces
**two linked documents** via `put_source()` — the untouched original (PDF
bytes, raw API response, pasted text) and the extracted text a claim's
indices are valid against (`SourceDocument.text_id` / `.original_id`).
They're linked, not merged, because `SourceSpan.document_id` must point at
the exact text indices were computed against, while a human wants to see
the original page; a parser upgrade can reprocess the original without
touching spans that point at old extracted text.

| Backend | Selected by | Notes |
|---|---|---|
| `LocalFileStore` | `STORAGE_BACKEND=local` (default) | Disk under `data/uploads/`, sidecar `_meta/*.json` links |
| `B2FileStore` | `STORAGE_BACKEND=b2` | Backblaze B2 via boto3's S3-compatible client; mirrors `LocalFileStore`'s layout (a `_meta/` key prefix, not S3 object-metadata headers) |

### `Repository` (`repository.py` + `models.py` + `db.py`)

Persistence for `Claim`, `Benchmark`, `Result`. `SqlAlchemyRepository` opens
a short-lived `Session` per method call — no table is created at import
time; Alembic migrations own schema creation against a real Neon Postgres
database (`DATABASE_URL`), or `Base.metadata.create_all()` against
in-memory SQLite in test fixtures.

One table per frozen schema (`claims`, `benchmarks`, `results`); nested
structures (`source_span`, `dimensions`, `gaps`, `generated`, `rate_band`)
stay as JSONB (`JSONB().with_variant(JSON(), "sqlite")` so the fast test
suite runs against plain JSON). `ClaimRecord`/`ResultRecord` share an
optional `result_id`/`run_id` — this is what lets `Repository.get_report()`
trace a persisted `Result` back to the exact `Claim`s that produced it, not
just "some claims exist somewhere."

A `pgvector` column (`BenchmarkRecord.benchmark_topics_embedding`, dim 384)
already exists for the eventual real semantic-coverage embeddings — not yet
populated or read by any code.

`run_pipeline` only persists when a `repository` is explicitly passed in
(the FastAPI app passes the real singleton; the ~250 existing tests that
call `run_pipeline` directly get zero DB access, by default).

## 9. Platform (`app/platform/`)

### Pipeline orchestrator (`pipeline.py`)

`run_pipeline(inp, benchmark_version, repository, on_status, run_id)` runs
all six stages in order (see §1's diagram) and assembles `Result`. Mints
`run_id` unconditionally up front — every `Result` is self-identifying
whether or not it's persisted. Accepts an externally-minted `run_id` too,
which is what lets the async `/analyze` endpoint know the id *before* the
pipeline finishes (see below).

### Async web flow (`api.py`)

`POST /analyze` does **not** run the pipeline inline — a real run makes
several live LLM calls and can take from seconds to (under Groq free-tier
rate-limit backoff) over a minute, sometimes much longer. Instead:

```
POST /analyze
     │  mint run_id, status_store.create(run_id)
     ▼
background_tasks.add_task(_run_pipeline_in_background, ...)
     │  303 redirect (not 307 — avoids re-POSTing the form)
     ▼
GET /analyze/{run_id}          → progress.html, polls...
GET /analyze/{run_id}/status   → JSON stage/detail/claims_found/...
     │  once stage == "done"
     ▼
GET /analyze/{run_id}/result   → result.html, reads from Repository
GET /analyze/{run_id}/claims   → claims.html, full per-claim audit trail
GET /report/{run_id}           → JSON: Result + Claims together
```

`_run_pipeline_in_background` is the one place that catches every exception
— there's no HTTP response left to raise into once the redirect has gone
out, so an unhandled error there would otherwise leave the progress page
spinning forever. It writes `stage="error"` to the status store instead.

### Status tracking (`status.py`)

`StatusStore` protocol, same spirit as `FileStore`/`Repository`. Backed
today by `InProcessStatusStore` — a dict behind a lock. **Known
limitation, by design:** a process restart mid-run loses the job and its
status together, which is coherent (no orphaned "still running" row
survives to mislead anyone) rather than a bug to fix in this slice.

`RunStatus.rate_limit_resume_at` is an absolute server-clock timestamp, not
"seconds remaining" — `/analyze/{run_id}/status` recomputes the remaining
seconds fresh against the server's own clock on every poll, so a slow poll
or a backgrounded browser tab never shows a stale or negative countdown.

`status_reporting(on_status)` (in `extractor.py`, wrapped around the whole
`run_pipeline` call) makes the callback available via a contextvar to code
that has no business importing the status store directly — including
`GroqClient`'s rate-limit retry loop running inside a `ThreadPoolExecutor`
worker (contextvars are explicitly propagated via `copy_context()` per
submission, since the executor does not do this automatically).

## 10. Configuration (`app/config/`)

### `weights.py`

Pure data, versioned by `CONFIG_VERSION`. Every number the scoring/gap-
ranking system uses lives here, not scattered through the logic:
`TIER_WEIGHTS`, `DIMENSION_WEIGHTS` (must sum to 1.0), the two caps,
`KEYWORD_REQUIRED_WEIGHT`/`KEYWORD_SEMANTIC_WEIGHT`, `EFFICACY_TABLE`,
`EFFORT_HOURS_TABLE`/`EFFORT_HOURS_FLOOR`, and the recency-decay constants
(`RECENCY_HALF_LIFE_DAYS`, `RECENCY_FLOOR`). `validate_config()` fails
loudly at FastAPI startup if weights are internally inconsistent (don't sum
to 1.0, or a table's keys don't match `DIMENSION_WEIGHTS`).

`recency_factor(observed_date, as_of=None)` — exponential decay from 1.0
toward `RECENCY_FLOOR`, halving every `RECENCY_HALF_LIFE_DAYS`; old-but-real
evidence never decays to zero.

### `env.py`

Loads `.env` exactly once, at import time, imported at the top of
`app/__init__.py` — so any entry point (server, script, test) has env vars
resolved before any other app code runs, regardless of import order.

## 11. The three swappable backends, side by side

| What | Interface | Env var | Free-tier default | Production option | File |
|---|---|---|---|---|---|
| LLM provider | `LLMClient` | `LLM_PROVIDER` | either key present | `anthropic` / `groq` | `ingestion/extractor.py` |
| File storage | `FileStore` | `STORAGE_BACKEND` | `local` (disk) | `b2` (Backblaze B2) | `storage/store.py` |
| PDF extraction | `PdfTextExtractor` | `PDF_PARSER` | `pypdfium2` | `docling` (layout-aware) | `ingestion/pdf_extractor.py` |

All three share two properties (see `CLAUDE.md`'s "Swappable backends"):
fail loudly at startup/import time on misconfiguration, never on first use;
and heavy production-only dependencies (`docling`, `boto3`) live in
`requirements-production.txt` or are imported lazily inside a constructor,
never at module top level — so the base install and the fast test suite
never need them.

## 12. Testing

Registered pytest markers (`pyproject.toml`) gate expensive/networked tests
out of routine iteration:

| Marker | What it exercises | Cost |
|---|---|---|
| (none) | Everything else — ~250 tests | Fast, offline |
| `slow` | Real Docling PDF conversion | Seconds; needs `requirements-production.txt` |
| `live_api` | Real Anthropic/Groq/B2 calls | Real quota; free-tier rate-limit backoff can run to ~26 minutes |
| `db` | Real Neon Postgres | Network; writes to the real hosted DB |

Day-to-day loop: `pytest -q -m "not slow and not live_api and not db"`.

## 13. What's built vs. what's next

Every pipeline stage in §1 is real today — the walking skeleton described in
`pipeline.py`'s own docstring has had every stub replaced. What's
deliberately deferred (see `CLAUDE.md`'s "After the slice works"):

- Wider ingestion — LinkedIn, portfolio sites, video, articles (schema
  already has the `SourceType` values; no parsers exist yet)
- Automated benchmark refresh (anchor monthly, radar daily — radar produces
  tags only, never weights); today's benchmark is one hand-written file
- Real semantic-coverage embeddings (the `pgvector` column exists,
  unpopulated; `score_keyword_coverage`'s semantic half is a placeholder
  synonym map)
- `case_study` / `proposal_draft` generated-asset kinds
- The presence engine (spec §9) and the learning loop (spec §7) — the
  learning loop is explicitly last: offline batch, ridge regression with
  non-negative coefficients, shrinkage toward the prior, and a human review
  gate before any weight change goes live — never automatic, never silent.
