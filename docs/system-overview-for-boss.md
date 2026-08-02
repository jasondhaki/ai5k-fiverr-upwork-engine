# How the Profile Intelligence System Works

*A plain-language walkthrough — no code, no jargon — for a management review.*

---

## 1. What this system does, in one sentence

A freelancer hands it their CV, their GitHub username, and their Upwork profile
text — and it tells them exactly how their profile compares to the people
actually winning work in their niche, and the fastest ways to close the gap.

Think of it as a coach with one very strict rule: it will never tell someone
to claim something they can't back up. Every statement it makes about a
person is traceable back to a specific line in their CV, a specific GitHub
repository, or a specific sentence in a client review. If it can't point to
where a claim comes from, it doesn't say it.

---

## 2. What goes in

| Input | How it's used |
|---|---|
| **CV / résumé** | Uploaded as a PDF. The system reads the actual text — work history, skills, dates. |
| **GitHub username** | Pulled automatically: repositories, languages used, how active and recent the work is. |
| **Upwork profile text** | Pasted in by the user. Existing bio, reviews, and job history are read for evidence. |

Nothing is scraped from a platform behind someone's back. The person supplies
their own data (uploads their own CV, pastes their own text); the system
never logs into Upwork or LinkedIn pretending to be them. That matters
because platforms ban scraping, and a flagged account would defeat the whole
purpose.

---

## 3. What happens in between (plain language)

```
1. Read everything &      2. Grade how           3. Compare to        4. Rank what        5. Draft an
   pull out claims    →      believable       →     what top       →     to fix first  →     improved
                             each claim is            earners do                              profile
```

### Step 1 — Read everything and pull out claims

The CV, GitHub, and Upwork text are each read separately (a CV needs
different handling than a GitHub profile), and every fact worth mentioning —
"built X," "worked at Y," "skilled in Z" — is pulled out as its own record,
along with exactly where it came from. If a claim can't be re-found
word-for-word in the original document, it's thrown out rather than guessed
at. This is the safeguard against the AI making something up: it's never
allowed to paraphrase a claim into existence — it must quote it, and the
system independently double-checks the quote is real.

### Step 2 — Grade how believable each claim is

Not all evidence is equal, so every claim gets sorted into one of eight
grades, from strongest to weakest:

| Strongest → Weakest | What it means |
|---|---|
| Client-verified result | A named outcome in a real client review — the hardest thing to fake |
| Project actually shipped | A live GitHub repo, app, or model people can go look at |
| Assessed by the system itself | Passed a structured skills check |
| Verified certification | Proctored and checkable with the issuer |
| Self-paced certificate | A badge from an online course, unverified |
| Past employer confirms it | Listed in real work history |
| Colleague recommendation | Someone else vouches for it |
| Just stated, no proof | The person simply says so — weakest, but still recorded |

The logic behind the ranking is simple and defensible: a shipped, working
project is harder to fake than a certificate, so it counts for more. A
profile built entirely on unproven, self-declared claims is automatically
capped at a low score — no amount of nice wording can push it higher. That's
a deliberate design choice, not a bug: it's what stops the tool from becoming
"just another way to inflate a résumé."

### Step 3 — Compare against what actually wins

Separately, the system keeps a running picture of what the *highest-earning*
profiles in a given specialty (e.g. "SMB workflow automation") actually look
like — how they title themselves, which words buyers search for, how many
portfolio pieces they show, what they charge. This picture is refreshed
monthly by studying top profiles and keeping only the patterns, not anyone's
personal data. A faster daily check watches for buzzwords rising or falling
out of fashion, but a human always reviews that before it's trusted.

### Step 4 — Rank what to fix first

Rather than a giant to-do list, the system scores each possible improvement
by "score gained per hour of effort" and surfaces a short, prioritized list —
plus anything that's a hard blocker (like an unproven claim) that must be
fixed before anything else, regardless of effort. It also makes sure the
list isn't just five quick five-minute wins; it always includes the one
change with the biggest overall payoff too, even if it takes longer.

### Step 5 — Draft an improved profile

Finally, it drafts a rewritten title, summary, and case studies using only
the proven claims. Every number that appears in the draft (e.g. "cut
processing time by 40%") is checked a second time against the original
source before it's allowed into the text. If nothing in a person's evidence
is strong enough to support a claim yet, the system leaves that section
blank and says so — it will not fill the gap with a plausible-sounding guess.

---

## 4. What the user actually sees

Deliberately, there is no single scary number shown up front. Research (and
simple experience) shows that leading with "you scored 4/100" makes people
feel judged and turns them off, even though the math behind it is fair.
Instead, the framing is: *"you have 11 claims and can currently prove 2 —
here's the fastest way to prove 3 more this week."* Same underlying number,
completely different experience.

```
PROFILE READINESS   41 / 100
Capped at 30 until claims are proven

DIMENSION      YOU NOW                     TOP TIER                    PRIORITY
Positioning    "Generic AI Developer"      Role + specialty + result   BLOCKING
Evidence       2 of 11 claims proven       Top tier proves most        BLOCKING
Keywords       Missing 4 required terms    Buyers search for these     3.2 pts/hr
Portfolio      1 item, no numbers          Three items, with numbers  1.8 pts/hr
Opening line   "I am a..."                 Leads with buyer's problem 1.5 pts/hr
Pricing        $25/hr                      Evidence supports $55-70   0.9 pts/hr
```

Below that, the report shows every individual claim and exactly which
document or repo it came from (full transparency, nothing hidden), which
skills the top tier has that this person doesn't yet (framed as "what to
learn next," not a penalty), and — once enough proof exists — a preview of
the rewritten Upwork profile itself, ready to copy in.

---

## 5. Why this is different from a keyword-matching tool

Most "resume scoring" tools just count how many buzzwords appear and reward
stuffing more of them in. That produces profiles that read unnaturally and,
more importantly, don't survive contact with a real client — an inflated
claim gets discovered in the first conversation and tanks the freelancer's
reputation score, which controls almost everything else on a platform like
Upwork.

This system is built around one non-negotiable rule instead: **if a claim
can't be traced back to a real source, it never appears on the profile.**
Every other design decision — the evidence grades, the score caps, the
"quote and verify" extraction — exists to enforce that one rule.

---

## 6. The architecture, without the jargon

It's one application built in clearly separated stages, like a factory line
— data goes in one end, a finished report comes out the other, and each
station only does one job:

- **Intake** — figures out what kind of file/source it's looking at and
  routes it to the right reader: a CV reader for PDFs, an API call for
  GitHub, plain text parsing for pasted Upwork text.
- **AI extraction** — a language model (Claude or an alternative) reads the
  text and pulls out structured facts — but only ever quotes, never invents.
- **Rules engine** — grading claims, scoring dimensions, and ranking gaps is
  all plain deterministic logic — no AI guesswork here, so results are
  consistent and explainable every time.

A few practical choices worth mentioning to a non-technical audience:

- **Swappable parts.** The AI model, the file storage, and the PDF reader
  are all built so any one of them can be swapped by flipping a setting —
  never by rewriting code. That keeps it cheap to run day-to-day and easy to
  upgrade for a paying customer without a rebuild.
- **Nothing is auto-published.** Every draft — profile rewrite, social post,
  proposal — is shown to the person for review. The system never posts or
  submits anything on someone's behalf.
- **The scoring "opinions" (how much each factor matters) are never changed
  automatically.** They start as sensible hand-set defaults and are only
  ever updated after enough real outcome data comes in, and even then a
  human has to sign off before it goes live — never a silent, automatic
  change.

---

## 7. Where it stands right now

**BUILT & WORKING** = already runs end-to-end today. **PLANNED NEXT** =
designed, not yet built.

| Stage | Status | What it covers |
|---|---|---|
| Reading the CV, GitHub, and Upwork text and pulling out claims | **BUILT** | The full read-and-extract pipeline, with the quote-and-verify safeguard |
| Grading each claim's evidence strength | **BUILT** | The eight-tier system described above |
| Comparing against a niche benchmark | **BUILT** | Reads a benchmark for one niche today; refreshing it automatically every month is next |
| Scoring the seven dimensions and ranking gaps | **BUILT** | Full scoring + prioritized gap list, including the caps that prevent inflated scores |
| Drafting the rewritten profile | **BUILT** | Titles, overview, case studies — all validated against real sources before showing them |
| Widening intake to LinkedIn, portfolio sites, video, articles | **PLANNED** | Same evidence pipeline, more sources feeding it |
| Automatic monthly benchmark refresh across many niches | **PLANNED** | Today the benchmark is a fixed, hand-loaded reference; this makes it self-updating |
| Auto-drafted LinkedIn/X posts and a content calendar | **PLANNED** | Ships after the core report is proven out — never auto-posts, always a human review step |
| Learning from real outcomes to fine-tune the scoring | **PLANNED** | Deliberately last — requires enough real users first, and always has a human approval gate |

**Bottom line:** the core loop — CV + GitHub + Upwork in, a graded,
prioritized, evidence-backed report out — already runs start to finish.
What's left is *widening* it (more input sources, more niches, more
automation of the benchmark) rather than proving the core idea works.
