# LLM Token & Request Optimization

Technical record of the Groq API cost-reduction work: what the extraction
pipeline used to send to the LLM, what it sends now, and why each change is
safe. For the general architecture, see [architecture.md](architecture.md);
for how extraction fits into the six-stage pipeline, see `CLAUDE.md`'s
"Replace stubs in this exact order," stage 1.

---

## 1. The problem

The free-tier Groq rate limit (tokens-per-minute and requests-per-minute)
was being exhausted quickly during real analysis runs. The root cause: the
extraction stage (`app/ingestion/extractor.py`) ran multiple independent LLM
"passes" per document — identity, history, skills — and **every pass
re-sent the entire document's full, unmodified text** as its prompt, with no
stripping and no length cap.

### Call count for a single analysis run (CV + GitHub[12 repos] + Upwork)

| Source | Passes (before) | Calls (before) |
|---|---|---|
| CV | identity, history, skills | 3 |
| GitHub (per repo × 12) | history, skills | 24 |
| Upwork | identity, skills | 2 |
| Generation (title + overview) | — | 2 |
| **Total** | | **31** |

GitHub dominated: each of up to 12 analyzed repositories cost 2 separate
calls, each one sending that repo's full README — badges, install
instructions, license text and all — with no truncation.

---

## 2. What changed, side by side

### 2.1 Pass merging (GitHub and Upwork)

**Before** — `extract_candidate_claims` ran every pass for a source through
a `ThreadPoolExecutor`, one `_run_pass` call per pass, each sending the same
`text` with a different system prompt:

```python
pass_names = _PASSES_BY_SOURCE.get(source_type, _DEFAULT_PASSES)
with concurrent.futures.ThreadPoolExecutor(max_workers=len(pass_names)) as pool:
    futures = {
        pool.submit(_run_pass, client, pass_name, text): pass_name
        for pass_name in pass_names
    }
```

For a GitHub repo (`history`, `skills`) this meant 2 calls, each carrying
the full README. For Upwork (`identity`, `skills`), 2 calls, each carrying
the full pasted text.

**After** — GitHub and Upwork now dispatch through one merged call whose
system prompt concatenates every relevant pass instruction verbatim:

```python
_MERGEABLE_PASS_SOURCES = frozenset({SourceType.GITHUB_REPO, SourceType.UPWORK_TEXT})

if source_type in _MERGEABLE_PASS_SOURCES and len(pass_names) > 1:
    label = "+".join(pass_names)
    batches = [(label, _run_pass(client, label, _merged_system_prompt(pass_names), text))]
else:
    # unchanged parallel-pass path, still used by CV
    ...
```

| | Before | After |
|---|---|---|
| GitHub, per repo | 2 calls, README sent twice | **1 call**, README sent once |
| Upwork | 2 calls, text sent twice | **1 call**, text sent once |
| CV | 3 calls | **3 calls (unchanged)** |

**Why this is safe for GitHub/Upwork but not CV:** a claim's evidence tier
is decided in `_provisional_tier` (the only place `evidence_tier` is set at
extraction time). That function branches on *which pass produced a claim*
in exactly one case: `source_type == CV and pass_name == "skills"` → forces
T8 (self-declared). For GitHub, every claim resolves to T2 regardless of
pass (`_PROVISIONAL_TIER_BY_SOURCE`); for Upwork, the one pass-independent
refinement is a text-pattern match against the grounded quote, not a
`pass_name` check. Merging GitHub/Upwork's passes therefore changes
*nothing* about how any claim is tiered. Merging CV's three passes would
require the model to self-report which category a claim belongs to — an
erosion of the explicit invariant that "the model that ran the pass never
supplies a tier" (`_ExtractedSpan` has no such field) — for a source that
only costs 3 calls total per run regardless of repo count. Not worth the
risk, so CV was left untouched.

Files: `app/ingestion/extractor.py` (`_MERGEABLE_PASS_SOURCES`,
`_merged_system_prompt`, `_run_pass`, `extract_candidate_claims`).

---

### 2.2 Rules-based boilerplate stripping

**Before:** the full derived document text (`_repo_text` for GitHub,
`extract_pdf_text`'s output for CVs) went straight into the LLM prompt,
badges/license/install-instructions and all.

**After:** a new `prepare_extraction_text` step runs first, removing:

- **Badge lines** — one or more markdown badge images (`![...](...)` or
  `[![...]](...)`) with nothing else on the line.
- **Known boilerplate sections** — License, Contributing, Table of
  Contents, Badges headings and everything under them up to the next
  heading (a conservative, explicitly curated list — not a generic
  "drop anything under an unimportant-looking heading" heuristic).
- **Generic install/run code fences** — a fenced code block is dropped only
  if *every* substantive line inside it (blank lines and `#`-comments don't
  count) matches the existing `_BOILERPLATE_COMMAND_PATTERN` (`npm run dev`,
  `yarn install`, etc.). A fence mixing real content with commands is left
  intact.
- **Known boilerplate phrases** — reuses the existing
  `_BOILERPLATE_PHRASES` list (fingerprints of `create-next-app`/
  `create-expo-app` default READMEs) that the post-hoc claim filter
  (`_is_boilerplate`) already relied on — the same list, applied one step
  earlier, not a second copy.

```python
def prepare_extraction_text(text: str) -> str:
    stripped = _strip_boilerplate_sections(text)
    return _cap_text(stripped, _extraction_char_limit())
```

This is deliberately **conservative by construction**: only text matching an
explicit, curated pattern is removed. Genuine content that shares no
vocabulary with any known boilerplate pattern is never touched — verified
directly (see §4): a clean README with no boilerplate at all comes back
byte-identical.

Files: `app/ingestion/extractor.py` (`_strip_boilerplate_sections`,
`_strip_boilerplate_command_fences`, `_BADGE_LINE_PATTERN`,
`_BOILERPLATE_SECTION_HEADING_PATTERN`).

---

### 2.3 Hard character cap

**Before:** no upper bound on prompt size — a large outlier document cost
tokens in direct proportion to its size, unbounded.

**After:** `_cap_text` truncates the (already-stripped) text to at most
`MAX_EXTRACTION_CHARS` characters (default **6,000** — chosen against this
codebase's own documented worst case, a 7,898-character README measured
during the original span-grounding work; see `CLAUDE.md`'s "Spec of
record"). Truncation prefers a paragraph or word boundary near the limit and
never appends synthetic marker text, since the returned string must stay a
literal, unmodified substring of the source for grounding to remain valid.

```python
DEFAULT_MAX_EXTRACTION_CHARS = 6000  # override via MAX_EXTRACTION_CHARS
```

Overridable via the `MAX_EXTRACTION_CHARS` environment variable, following
the same fail-loudly-on-a-bad-value convention as the existing
`GITHUB_REPO_CAP`.

---

### 2.4 Consistency requirement (why both call sites changed together)

`prepare_extraction_text`'s output is passed to **both**
`file_store.put_source(text=...)` **and** `extract_candidate_claims(text=...)`
— never the raw text to one and the prepared text to the other. A claim's
span indices are computed against whichever string was actually sent to the
LLM; storing a *different* string under that same `document_id` would make
`reverify_span` (`app/generation/validator.py`) fail the next time that
document is read back during asset generation. Both call sites
(`app/ingestion/github_parser.py`, `app/ingestion/cv_parser.py`) were
updated together for this reason.

The trivial-repo skip check (`MIN_REPO_TEXT_CHARS`) also moved to run
**after** stripping rather than before, so a repo whose raw text only clears
the floor because it's padding of badges/install-instructions is now
correctly treated as trivial too — one skip check, not two.

---

## 3. Net effect on the 31-call baseline

| Source | Before | After |
|---|---|---|
| CV | 3 calls, full text | 3 calls, stripped + capped text |
| GitHub (12 repos) | 24 calls, full README ×2/repo | **12 calls**, stripped+capped README ×1/repo |
| Upwork | 2 calls, full text ×2 | **1 call**, full text ×1 |
| Generation | 2 calls (unchanged, already minimal) | 2 calls |
| **Total** | **31** | **18 (~42% fewer requests)** |

Every merged call also stops re-sending the same document text twice, and
every GitHub/CV call now carries less text per call on top of that — so the
token-volume (TPM) reduction is larger than the request-count (RPM)
reduction alone suggests.

---

## 4. Measured impact (synthetic fixtures)

Live verification against a real GitHub profile (`torvalds`) was attempted
but blocked by the sandbox's shared, unauthenticated GitHub API rate limit
being exhausted at the time. As a substitute, the same reduction was
measured directly against realistic README fixtures (the actual
`create-next-app`/`create-expo-app` default READMEs, each with one genuine
project sentence appended — the same fixtures `tests/test_extractor.py`
uses) with a call-counting fake LLM client (zero cost, zero network calls):

| Repo | Raw chars | Prepared chars | Reduction |
|---|---|---|---|
| Next.js boilerplate README | 873 | 466 | 47% |
| Expo boilerplate README | 859 | 295 | 66% |
| Clean README (no boilerplate) | 194 | 194 | **0%** |

```
LLM calls    -> before: 6   after: 3   (50% fewer)
Prompt chars -> before: 3,852   after: 955   (75% fewer)
```

The third row is the important correctness signal: a README with no
boilerplate at all is returned **unchanged**, confirming the stripper only
removes recognized patterns and never erodes genuine content.

---

## 5. Test coverage added

All in `tests/test_extractor.py` (fast suite, no network/API calls):

- Merged-call behavior: call counts drop from 2→1 for GitHub and Upwork
  fixtures; the merged system prompt still carries both original pass
  instructions verbatim.
- Three existing dedup tests (`test_exact_duplicate_claim_text_collapses_to_one`,
  `test_near_duplicate_reworded_claim_collapses_to_one`,
  `test_distinct_real_accomplishments_are_never_merged`) were rewritten to
  simulate a single merged-call response instead of two per-pass responses —
  one of the three was silently exercising the wrong behavior after the
  merge (still passing, but no longer testing what it claimed to) and has
  been corrected along with the other two for consistency.
- Boilerplate stripping: badge-only lines, License/Contributing sections
  (including that a dropped section correctly stops at the next heading),
  a generic install fence with interspersed `# or` comment lines, and a
  mixed fence (real content + commands) that must survive intact.
- Character cap: short text left untouched; long text truncated to a
  literal prefix; `MAX_EXTRACTION_CHARS` env override, including its
  fail-loudly behavior on a non-positive or non-integer value.
- End-to-end sanity check: a claim quoted from the surviving portion of a
  stripped-and-capped document still grounds correctly.

Full fast suite (`pytest -q -m "not slow and not live_api and not db"`):
**313 passed**, 0 failed.

---

## 6. Considered and deliberately not done

| Idea | Status | Why |
|---|---|---|
| **CV pass merging** | Excluded | Tier-determinative (`_provisional_tier`'s CV/skills branch); CV is already the cheapest source (3 calls/run, not multiplied by repo count) |
| **GitHub repo batching** (multiple repos in one call) | Rejected | Only reduces request count (RPM), not token volume (TPM) — the merged calls above already address both. Would also require a multi-document grounding variant, risking a claim's quote grounding against the wrong repo — a real weakening of `span_grounding.py`'s "core invariant" for a technique that doesn't even cut tokens |
| **Response cache** (skip identical repeat calls) | Deferred | Real ROI is mostly dev-loop/demo re-runs, not steady-state production traffic (real CVs/READMEs are rarely resubmitted byte-identical) |
| **Smaller/faster Groq model** (e.g. `llama-3.1-8b-instant`) | Deferred | Trades directly against extraction quality — needs an empirical before/after comparison (claim yield, grounding rate) before ever becoming the default, not a blind swap |
| **Embedding-based chunk selection** ("RAG-style" pre-filtering) | Not applicable / not worth it yet | Classic RAG (retrieval over an external corpus) doesn't apply — there is no external corpus; each call's only input is the one document being analyzed. Self-document chunk selection is technically possible but unnecessary at this codebase's actual document sizes (worst measured case: 7,898 characters) — the rules-based stripper above achieves the same goal with zero new dependency and full explainability |

---

## 7. Configuration reference

| Env var | Default | Purpose |
|---|---|---|
| `GITHUB_REPO_CAP` | 12 | How many most-recently-pushed repos are analyzed at all (pre-existing) |
| `MAX_EXTRACTION_CHARS` | 6000 | Hard cap on characters sent per extraction call, after boilerplate stripping (new) |

Both fail loudly at read time on a non-positive or non-integer value, per
this codebase's standing convention for env-var-selected knobs (see
`CLAUDE.md`'s "Swappable backends").
