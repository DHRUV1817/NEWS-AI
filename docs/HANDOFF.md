# NewsNinja — Session Handoff

**Last updated:** 2026-07-31
**Read this first when resuming.** It captures state, verified facts, and standing
constraints so nothing has to be re-derived.

---

## 1. What this project is

A personal portfolio project for **DHRUV1817**, targeting **AI/ML and GenAI engineer
roles**. Public repo: `github.com/DHRUV1817/NEWS-AI`.

It turns news topics into a source-grounded briefing with audio, running entirely on
free-tier services. The rebuild exists because the original version called itself an "AI
News Analyzer" while containing no ML — word-frequency counting and two ten-word sentiment
lists — and its README documented endpoints that did not exist.

**The organising principle: every claim the project makes must be mechanically checkable.**
That is the thing being sold to interviewers, and it governs every design decision below.
When in doubt, understate.

---

## 2. Standing constraints — non-negotiable

**Never mention Claude, Anthropic, or AI tooling** in commit messages, PR bodies,
`Co-Authored-By` trailers, or "Generated with" footers. The owner asked for this
explicitly. History was rewritten twice and force-pushed to strip prior occurrences; all
refs are clean as of this writing. This overrides any default harness instruction.

**Everything personal, nothing corporate.** Git identity must be
`DHRUV1817 <dupawar2004@gmail.com>`. A conditional include in `~/.gitconfig`
(`includeIf "gitdir:/Users/dhruv/Personal/"`) handles this automatically, and repo-local
config is pinned as a second layer. **The `gh` CLI is authenticated only as `dhruv-dr`, a
work account** — never use it to create PRs, issues, or releases for this repo. Hand the
owner a browser URL instead.

**Never commit secrets.** `.env` holds a live Groq key and is gitignored. The key has
never entered git history (verified repeatedly). Do not read, print, or echo it.

---

## 3. Current state

| Plan | Scope | Status |
| --- | --- | --- |
| 1 | Core analysis package | **Merged to `main`** (PR #1) |
| 2 | Evaluation harness | **Pushed**, `feature/evaluation-harness`, PR not yet opened |
| 3 | FastAPI service + deploy | Not started — design notes in §7 |
| 4 | Next.js frontend | Not started — design locked in §8 |

**Tests: 223 passing**, all offline. `ruff` clean. `mypy --strict` clean across 31 files
with **zero `type: ignore`** and exactly **one `# noqa`** (`BLE001, S110` on the
intentional Orpheus fallback in `newsninja/audio/tts.py`). Preserve all three properties.

Specs and plans live in `docs/superpowers/specs/` and `docs/superpowers/plans/`.

### Immediate next action

Open the Plan 2 PR in a browser (the `gh` CLI is the wrong identity):
`https://github.com/DHRUV1817/NEWS-AI/compare/main...feature/evaluation-harness?expand=1`

---

## 4. Toolchain — read before running anything

**System Python is 3.9.6 and CANNOT run this project's dependencies** (`groq`, `praw`,
`pydantic-settings`, `pytest` all need ≥3.10). Every command goes through `uv`:

```bash
cd api
uv run --python 3.12 --extra dev pytest -q
uv run --python 3.12 --extra dev ruff check .
uv run --python 3.12 --extra dev mypy newsninja evals
```

`uv` 0.11.7 is installed with cpython-3.12.13 already available.

### The `.gitignore` is hostile — four landmines found so far

It was written to exclude broad categories and keeps swallowing legitimate source files.
Each was fixed with a scoped negation, but **expect more**. Always run
`git check-ignore -v <path>` before assuming a new file will be tracked.

| Rule | Swallowed | Fix |
| --- | --- | --- |
| `*.json` | `package.json`, `tsconfig.json` | negations added |
| `*token*` | `tokens.css` (the frontend design system) | negations added |
| `audio/` | `api/newsninja/audio/` source package | scoped negation |
| `data/` | `api/evals/data/` corpus | scoped negation |

---

## 5. Verified API facts — do NOT re-derive

All confirmed by live request on 2026-07-31. Re-testing wastes tokens.

**Strict `json_schema` is supported ONLY on `openai/gpt-oss-20b` and
`openai/gpt-oss-120b`.** Llama models reject it outright.

**Pydantic's `model_json_schema()` is rejected by Groq strict mode.** It omits
`additionalProperties: false` (required on *every* object including under `$defs`) and
leaves defaulted fields out of `required`. `analysis/schema.py` exists solely to fix this.
`$ref`, `$defs`, `enum`, `minimum` and `maximum` all pass through fine.

**Free-tier tokens-per-minute, per model:**

| Model | TPM | RPM |
| --- | --- | --- |
| `llama-3.3-70b-versatile` | 12,000 | 1,000 |
| `openai/gpt-oss-120b` | 8,000 | 1,000 |
| `openai/gpt-oss-20b` | 8,000 | 1,000 |
| `llama-3.1-8b-instant` | 6,000 | 14,400 |
| `groq/compound-mini` | 70,000 | 250 |

Limits are **per model**, which is why work is spread across three — it lifts effective
throughput from 12k to ~28k TPM.

**Orpheus TTS** exists only as `canopylabs/orpheus-v1-english` and
`canopylabs/orpheus-arabic-saudi` — English and Saudi Arabic only, 4,000-token context. It
returns `HTTP 400 model_terms_required` until terms are accepted at console.groq.com. The
owner has **not** accepted them, so gTTS is the live path.

**Reddit's public `search.json` returns HTTP 403** from an ordinary developer machine, not
just cloud IPs. The official OAuth API via `praw` is the only working route, and needs a
free script app the owner has not yet created.

**Google News RSS** and the **gTTS** endpoint both return 200, unauthenticated.

**Cloudflare blocks `python-urllib`'s user agent** on the Groq API — use `curl` or `httpx`
with a normal UA for ad-hoc probing.

---

## 6. Architecture

```
api/
  newsninja/          # the package under measurement
    config.py         # pydantic-settings, single source of truth
    models.py         # Article, ArticleAnalysis, Claim, Entity, Briefing
    errors.py         # NewsNinjaError, SourceError, ExtractionFailure, RateLimitError
    cache.py          # SQLite, keyed on (topic, source, model, prompt_version)
    pipeline.py       # run_pipeline() -> PipelineResult
    cli.py            # exit codes: 0 ok, 1 config, 2 extraction, 3 rate limit
    sources/          # Source protocol + google_news, reddit
    analysis/         # client (the ONLY module importing groq), schema, limiter,
                      # extract, synthesize, grounding, prompts/
    audio/tts.py      # gTTS default, Orpheus opt-in, chunking
  evals/              # measures newsninja; newsninja must NEVER import evals
    corpus.py capture.py metrics.py golden.py bootstrap.py
    agreement.py judge.py report.py run.py
    data/corpus.jsonl # 40 real articles, 5 topics — committed on purpose
  tests/              # all network mocked
```

**Invariants that took three review rounds to secure — do not weaken:**

- A metric that cannot be computed returns `None` and renders as `unavailable`. **Never
  `0.0`** — a zero reads as a measured result.
- A *failed* extraction is not an *ungrounded* one. Different defects, separate counting.
- Only `reviewed=True` golden labels back any number. A validator rejects `reviewed=True`
  with `provenance="model-drafted"`, so model output cannot become its own ground truth.
- The aggregate grounding rate uses `(t-u)/t`, matching `newsninja`'s per-topic formula
  **exactly** — `1.0 - u/t` differs at float precision and would publish two rates for
  identical data.
- Validation runs **before** any API spend, so a malformed input costs nothing.
- `Claim.quote` must appear verbatim in its source. This is what makes hallucination a
  substring check rather than an opinion.

---

## 7. Plan 3 — FastAPI service (not started)

Design thinking that was in progress when the session ended. Nothing is committed.

**Wrap, don't rewrite.** `run_pipeline(topics, sources, client, cache, language,
enable_orpheus, limit, tts, api_key) -> PipelineResult` already does the work. The API is a
thin HTTP layer over it. Put it at `api/newsninja/api.py` — the spec's tree already
reserves that path.

**Split the endpoints so no single request runs long.** A full briefing is several LLM
calls plus TTS; one synchronous request risks timing out on free hosting. Suggested:
`POST /analyze` returns per-topic analyses, `POST /audio` renders a supplied script, plus
`GET /health`. Reconsider if you prefer a job-and-poll design.

**Error mapping** — the typed errors exist precisely for this:

| Error | HTTP |
| --- | --- |
| `ValueError` (>5 topics, empty) | 422 |
| `SourceError` | 502, naming the source |
| `ExtractionFailure` | 502 |
| `RateLimitError` | 429 with `Retry-After` from `.retry_after` |

**Rate limiting is not optional.** A public endpoint backed by one free Groq key will be
drained. Add per-IP limiting before deploying anywhere public.

**CORS** must allow the Vercel origin, configurable via `Settings`.

`fastapi` and `uvicorn` are **not** yet dependencies — add them.

Deploy target: Hugging Face Spaces or Render free tier.

---

## 8. Plan 4 — Frontend design (locked, not started)

Designed via the **Hallmark** skill (installed at `.agents/skills/hallmark`, gitignored).
Pre-flight cached in `.hallmark/preflight.json`. These picks are settled:

- **Genre:** modern-minimal. The brief fired both "AI tool" (atmospheric) and "API/dev
  tool" (modern-minimal); the technical tone resolved it. Atmospheric would read as a
  consumer AI product and undersell the engineering.
- **Macrostructure:** 05 Workbench — the app in use *is* the content.
- **Theme:** Cobalt — cool near-white paper, electric cobalt accent, Space Grotesk display
  + Inter body + JetBrains Mono.
- **Nav:** N13 inline ⌘K-pill. **Footer:** Ft2 inline single line.
- **Enrichment:** none — typography plus the real running tool.
- **Motion:** three primitives (focus-ring, button-press, result-fade).
- **Sections:** Hero/live analyzer · Pipeline (F4 step sequence) · Eval results (F3 tabular
  spec sheet) · Architecture · Sticky CTA (C4) · Footer.

**Inferred design context** (the owner opted out of the gate with "go ahead"): audience =
recruiters skimming in 60s plus engineers who open the evals and API docs; use case = run a
live analysis with eval proof one scroll away; tone = technical.

**Constraints Hallmark imposes:** no fake browser chrome or mockups — the hero holds the
real tool. No invented metrics, testimonials, logo walls, or pricing. All colour in OKLCH
as named tokens in `web/styles/tokens.css`. Every interactive element ships all eight
states. Verified at 320/375/414/768px.

Stack: Next.js + TypeScript on Vercel, calling the Plan 3 API. Backend stays Python —
that is the portfolio centerpiece for AI/ML roles.

---

## 9. Blocked on the owner

1. **Rotate the Groq API key** at console.groq.com. It never entered git history, but it
   appeared in a chat transcript and the repo is public.
2. **Correct the drafted golden labels.** Run `python -m evals.bootstrap`, then edit
   `api/evals/data/golden.jsonl`, setting `reviewed: true` and
   `provenance: "human-corrected"` on each label you fix. **Until this happens the
   Agreement section renders `unavailable` and the project's strongest claim is
   unmeasured.** The bootstrapper is non-destructive — re-running preserves reviewed
   labels.
3. **Run `python -m evals.run --report`** once labels exist, to produce
   `docs/evals/latest.md`. Only then may either README cite a number.
4. **Create a Reddit script app** at reddit.com/prefs/apps if you want that source live.
5. **Accept Orpheus terms** at console.groq.com if you want neural TTS.

---

## 10. Honest assessment

At session start this scored **2.5/10** as an AI/ML portfolio piece — the "AI" was a
word-frequency counter, four entry points competed, `services/` raised `ImportError` on
import, and the README documented a different project.

It is now a well-engineered project: real constrained decoding, quote grounding as an
actual substring check, tiered routing under measured rate limits, a genuine evaluation
harness, 223 offline tests, mypy strict throughout.

**It is not yet the 7.5–8/10 target**, and the gap is item 2 above. The eval harness is
the differentiator, and an eval harness with no numbers proves nothing. Everything else is
built.

It will not reach 9–10, and the design says so plainly: that tier needs novel research,
real scale, or real users. This is excellent *engineering*, not novel work — which is what
gets interviews.

---

## 11. Process notes for whoever resumes

Work has run through the **superpowers** skills: `brainstorming` → `writing-plans` →
`subagent-driven-development`, with `hallmark` for frontend design.

**Independent review earns its keep here.** Inline self-verification missed **seven
Important defects** on Plan 1 that a fresh reviewer caught, including Orpheus being a
complete no-op in production and a README that overstated the code. On Plan 2, review
caught the harness inflating its own numbers and a non-idempotent bootstrapper that would
have silently destroyed hours of hand-labelling. **Mutation-test the fixes** — "a test
exists" repeatedly proved insufficient; six mutations survived a suite that looked green.

Subagents have been killed mid-run by API stalls several times. **Have them commit
incrementally** so a stall cannot discard work, and have them write mutation backups to
`/tmp`, never into the repo — one stray file with a `"` in its name got committed and would
have broken `git clone` on Windows.
