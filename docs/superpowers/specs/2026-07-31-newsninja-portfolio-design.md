# NewsNinja — Portfolio Rebuild Design

**Date:** 2026-07-31
**Status:** Awaiting review
**Target:** Portfolio piece for AI/ML and GenAI engineer roles

---

## 1. Why this rebuild exists

The current repository presents itself as an "AI News Analyzer." It is not one.

`create_smart_summary` counts word frequencies. `analyze_sentiment_advanced` compares a
ten-word positive list against a ten-word negative list. The single genuine model call
(`utils.py:51`) targets `api-inference.huggingface.co`, a deprecated endpoint, and
silently falls back on failure. The README documents `POST /analyze` and
`POST /generate-audio`; neither exists. It claims caching and sentiment charts; neither
was wired up.

Beyond the AI gap, the repository carries four competing entry points
(`streamlit_app.py`, `single_file.py`, `backend.py` + `frontend.py`, `start.py`), one of
which launches two files that were never committed. `services/news_service.py` imports
`TopicAnalysis` from `models.py`, which does not define it, so the entire `services/`
package raises `ImportError`. `config.py` is imported by nothing.

The gap between claim and code is the problem this rebuild solves. An interviewer who
opens one file and finds a keyword list labelled "advanced sentiment analysis" reads the
rest of the candidate's work more skeptically. The fix is not more features. It is making
every claim in the repository mechanically checkable.

### Verified facts this design rests on

All confirmed by live request on 2026-07-31, not assumed:

| Check | Result |
| --- | --- |
| Groq API key | Works |
| `llama-3.3-70b-versatile` | 12,000 TPM · 1,000 RPM |
| `openai/gpt-oss-120b` | 8,000 TPM · 1,000 RPM |
| `openai/gpt-oss-20b` | 8,000 TPM · 1,000 RPM |
| `llama-3.1-8b-instant` | 6,000 TPM · 14,400 RPM |
| `groq/compound-mini` | 70,000 TPM · 250 RPM |
| Strict `json_schema` support | **Only** `openai/gpt-oss-20b` and `openai/gpt-oss-120b` |
| Orpheus TTS | `HTTP 400 model_terms_required` — free, needs a terms click |
| Orpheus languages | English and Saudi Arabic only; 4,000-token context |
| Google News RSS | `200` |
| Reddit public `search.json` | **`403`** — from the developer's own machine |
| gTTS endpoint | `200` |

Two of these overturn assumptions the current code makes. Reddit's public JSON is already
blocked, so `reddit_scraper.py` is broken today rather than merely fragile in deployment.
And the Llama models cannot do constrained decoding, which determines model routing.

A measurement worth keeping: asked for the same extraction, `llama-3.3-70b-versatile` in
loose `json_object` mode returned `{"entities": "Apple", "stance": "Positive"}` — a string
where an array belongs, and a value outside the declared enum. `openai/gpt-oss-20b` under
strict schema returned `{"entities": ["Apple","iPhone","China"], "stance": "positive"}`.
This is reproducible evidence for the constrained-decoding decision and belongs in the
README.

---

## 2. Scope

**In scope:** Python analysis package, FastAPI service, evaluation harness, Next.js
frontend, tests, CI, deployment, documentation rewrite.

**Retained from the current product** (user decision): audio briefings, Reddit as a second
source, multi-language support, sentiment analysis. Each is rebuilt rather than preserved.

**Deleted** (all paths are repository-root, as they exist today): `backend.py`,
`frontend.py`, `single_file.py`, `start.py`, `streamlit_app.py`, `utils.py`,
`news_scraper.py`, `reddit_scraper.py`, `services/`, `config.py`, `models.py`,
`ai-journalist.pdf`.

Note that `api/newsninja/config.py` and `api/newsninja/models.py` in section 3 are new
files inside the new package. They share names with deleted root-level files but no
content — the root-level originals are removed outright, not moved.

That is every current file. The honest reading is that the value here is the product idea
and the choice of data sources, not the existing implementation. Working logic from
`streamlit_app.py` is ported into the new package rather than preserved in place.

**Explicitly out of scope:** user accounts, persistence beyond the cache, real-time
streaming updates, mobile apps, paid infrastructure of any kind.

---

## 3. Architecture

```
api/                        # Python 3.11+
  newsninja/
    __init__.py
    config.py               # pydantic-settings; single source of truth
    models.py               # Article, ArticleAnalysis, Claim, Entity, Briefing
    sources/
      base.py               # Source protocol: fetch(topic) -> list[Article]
      google_news.py        # RSS, no auth
      reddit.py             # official OAuth API via praw
    analysis/
      client.py             # Groq wrapper: schema calls, retries, budget, accounting
      extract.py            # Article batch -> ArticleAnalysis (typed)
      synthesize.py         # analyses -> Briefing
      translate.py          # Briefing -> target language
      prompts/              # versioned prompt files + PROMPT_VERSION
    audio/
      tts.py                # gTTS default; Orpheus opt-in; chunking
    cache.py                # SQLite, keyed on (topic, source, model, prompt_version)
    api.py                  # FastAPI app
  evals/
    golden/articles.jsonl   # ~40 hand-labelled fixtures
    metrics.py              # P/R/F1, grounding, schema validity, Cohen's kappa
    judge.py                # rubric-scored LLM judge
    run.py                  # CLI: python -m evals.run --report
  tests/                    # pytest; all network mocked

web/                        # Next.js 15 + TypeScript
  app/
  components/
  styles/tokens.css         # Hallmark Cobalt tokens

docs/
  evals/                    # committed eval reports
  superpowers/specs/
```

**Why the frontend is TypeScript and the backend stays Python.** The analysis layer,
Pydantic schemas, and eval harness are the portfolio centerpiece for AI/ML roles, and
Python is the language those reviewers expect to read. The frontend is TypeScript because
Streamlit cannot deliver the UI quality this project needs — it does not expose DOM
control, so nav structure, section rhythm, and component styling are all unavailable.

**Why the package boundary matters.** It is not cosmetic. It is what lets the eval harness
run in CI without booting a web server, and what lets `pytest` reach the analysis code
directly. The `Source` protocol in `sources/base.py` means adding a source is a new file
rather than an edit to the pipeline, and makes sources trivially mockable.

---

## 4. Free-tier stack

Every component verified free, no credit card:

| Layer | Choice | Cost |
| --- | --- | --- |
| LLM | Groq free tier | Free |
| News | Google News RSS | Free, no auth |
| Reddit | Official API via `praw`, script app | Free, OAuth |
| TTS (default) | gTTS | Free, no auth |
| TTS (upgrade) | Orpheus on Groq | Free after terms acceptance |
| Translation | Groq | Free |
| Eval judge | Groq | Free |
| Frontend host | Vercel | Free |
| Backend host | Hugging Face Spaces or Render | Free |
| CI | GitHub Actions (public repo) | Free |
| Cache | SQLite on disk | Free |

### The rate limit is the engineering story

Five topics at eight articles each is roughly 32,000 tokens — nearly three times the 70b
model's per-minute ceiling. A naive per-article loop throttles or fails. Three mechanisms
handle it, and each is something an AI engineer gets asked about in interviews:

**Tiered routing.** Limits are enforced per model, so spreading stages across models
multiplies effective throughput from 12,000 to roughly 28,000 TPM.

| Stage | Model | Rationale | TPM |
| --- | --- | --- | --- |
| Per-article extraction | `openai/gpt-oss-20b` | Cheapest with strict schema support | 8,000 |
| Briefing synthesis | `openai/gpt-oss-120b` | Strongest reasoning, strict schema | 8,000 |
| Translation and script prose | `llama-3.3-70b-versatile` | No schema needed, highest budget | 12,000 |
| Eval judge | `openai/gpt-oss-120b` | Strict schema for score objects | 8,000 |

**Batching.** Articles are grouped per topic into one structured call rather than one call
each — five requests instead of forty.

**Token-budget limiter.** A sliding-window limiter reads Groq's
`x-ratelimit-remaining-tokens` response header and waits pre-emptively rather than
retrying into failure.

Paired with the SQLite cache, a repeat demo run costs zero tokens. This is not a nicety:
without caching, the eval suite would exhaust the free tier on every CI run.

---

## 5. AI core

`analysis/client.py` wraps Groq with four responsibilities: strict-schema calls,
validation-feedback retries, the token-budget limiter, and usage accounting. Nothing else
in the package touches the Groq SDK directly, so swapping providers or mocking in tests
changes one file.

Pydantic models are the contract:

```python
class Entity(BaseModel):
    name: str
    kind: Literal["person", "org", "place", "product", "other"]

class Claim(BaseModel):
    text: str
    quote: str          # MUST appear verbatim in the source article

class ArticleAnalysis(BaseModel):
    topic: str
    summary: str
    entities: list[Entity]
    stance: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_claims: list[Claim]
```

The `Claim.quote` constraint is the load-bearing design decision. Requiring every claim to
carry a verbatim source span makes hallucination *mechanically detectable* — a string
containment check, not a matter of opinion. This is what the grounding metric in section 6
measures.

**Retry policy.** On a Pydantic validation failure, the error text is fed back into the
next attempt as a correction message rather than blindly resampling. Capped at two
retries, after which a typed `ExtractionFailure` propagates and the UI degrades visibly
rather than silently.

**Prompt versioning.** Prompts live in `analysis/prompts/` as versioned files with a
`PROMPT_VERSION` constant that participates in the cache key. Changing a prompt invalidates
exactly the affected cache entries, and evals can compare versions against each other.

### Multi-language, done honestly

The current implementation passes a language code to gTTS while the analysis stays in
English. Keeping the feature means actually translating: the briefing is translated by
`llama-3.3-70b-versatile` before TTS.

Orpheus covers English and Saudi Arabic only, so TTS routes by language — Orpheus for `en`
and `ar` when the user has accepted terms, gTTS for the other ten. The README states this
split rather than implying uniform neural quality.

### Reddit, working

The public JSON endpoint returns 403. The rebuild uses the official API via `praw` with a
free script app (client ID + secret). Credentials are optional: absent them, the Reddit
source is disabled and the UI says so rather than showing a silent failure.

---

## 6. Evaluation harness

This is the centerpiece. It is what separates this project from every other news
summarizer, and it is the part an interviewer will probe hardest.

`evals/golden/articles.jsonl` holds roughly 40 hand-labelled articles: entities, stance,
and which claims are genuinely supported by the source. This requires 2–3 hours of the
developer's own labelling effort and cannot be delegated to a model without destroying its
value as ground truth.

Three metric families, deliberately separated by how objective each is:

**Deterministic.** Schema validity rate, entity precision / recall / F1 against the labels,
quote-grounding rate (does every `Claim.quote` appear in the source?), and hallucination
rate. No model judges these — they are arithmetic.

**Agreement.** Stance accuracy and Cohen's kappa against the labels. Kappa rather than raw
accuracy because stance is subjective and chance agreement should be discounted.

**Judged.** An LLM scores summary quality on a rubric of coverage, neutrality, and
coherence. Reported separately from the deterministic metrics and calibrated against a
human-scored subset, so the report can state the correlation rather than asserting the
judge is trustworthy.

`python -m evals.run --report` emits a Markdown table committed to `docs/evals/`. The
README cites measured numbers, never adjectives.

**A comparison worth running and publishing:** the same golden set scored under strict
schema (`gpt-oss-20b`) versus loose `json_object` (`llama-3.3-70b`). The preliminary
finding in section 1 suggests a large schema-validity gap. Publishing it turns a design
decision into a defended one.

---

## 7. Frontend design

Designed via the Hallmark skill. Pre-flight found no existing design system to preserve.

- **Genre** · modern-minimal. The brief fired both "AI tool" (atmospheric) and "API /
  developer tool" (modern-minimal); the technical tone resolved it. Atmospheric would read
  as a consumer AI product and undersell the engineering.
- **Macrostructure** · 05 Workbench — the app in use is the primary content.
- **Theme** · Cobalt — cool near-white paper, electric cobalt accent, Space Grotesk display
  + Inter body + JetBrains Mono for code and metrics.
- **Nav** · N13 inline ⌘K-pill · **Footer** · Ft2 inline single line.
- **Enrichment** · none. Typography plus the real running tool.
- **Motion** · three primitives: focus-ring, button-press, result-fade. The genre composes
  rather than reveals.

**Sections in DOM order:** Hero / live analyzer · Pipeline (F4 step sequence) · Eval
results (F3 tabular spec sheet) · Architecture · Sticky CTA (C4) · Footer.

**Inferred design context** (user opted out of the gate with "go ahead"): audience =
recruiters skimming in 60 seconds plus engineers who will open the eval results and API
docs; use case = run a live analysis, with eval proof one scroll away; tone = technical.

### Constraints held

No fake browser chrome, phone frames, or mock IDE windows — the hero contains the real
working tool, which satisfies Hallmark gate 47 by construction rather than by omission.

No invented metrics, testimonials, logo walls, or pricing tiers. A portfolio project has no
customers, and fabricating them is precisely the tell this rebuild eliminates. The eval
section renders labelled `—` placeholders until the harness produces real numbers.

All colours in OKLCH as named tokens in `web/styles/tokens.css`; no inline colour values.
Every interactive element ships all eight states. Verified at 320 / 375 / 414 / 768 px.

---

## 8. Error handling

Failures are typed and surfaced, never swallowed. The current code returns strings like
`"News temporarily unavailable"` from inside exception handlers, which makes failure
indistinguishable from a legitimate empty result.

| Failure | Behaviour |
| --- | --- |
| Source returns nothing | Typed empty result; UI shows "no articles found", distinct from an error |
| Source raises | `SourceError` captured per source; other sources still run; UI names the failed source |
| Schema validation fails | Retry with validation feedback, max 2; then `ExtractionFailure` surfaced |
| Rate limit approached | Limiter waits pre-emptively; UI shows a queued state |
| Rate limit hit anyway | Typed `RateLimitError` with retry-after; UI shows the wait |
| Orpheus terms not accepted | Detected on first call; falls back to gTTS and says so once |
| Reddit credentials absent | Source disabled at startup; UI shows it as unavailable, not broken |

No bare `except:`. No exception handler that returns a user-facing string in place of
raising.

---

## 9. Testing

`pytest`, with all network access mocked through recorded fixtures so CI is fast, free, and
offline.

- **Unit** — each source's parser against recorded payloads; cache key construction and
  invalidation; the token-budget limiter against synthetic header sequences; TTS language
  routing; schema validation and the retry-feedback path.
- **Integration** — full pipeline against mocked sources and a stubbed Groq client,
  asserting typed outputs and correct degradation on each failure mode in section 8.
- **Eval** — run on demand rather than per-commit, to respect the free tier.

GitHub Actions runs `ruff`, `mypy`, and `pytest` on push. The eval suite is a separate
manually-triggered workflow.

---

## 10. Documentation

The README is rewritten to undersell. What it does, the measured eval numbers, the
architecture diagram, a screenshot, the live URL, and an explicit limitations section
covering the free-tier rate ceiling, the Orpheus language restriction, and the fact that
sentiment is a model judgement rather than ground truth.

A `LICENSE` file is added — MIT, matching the claim the current README already makes
without one.

---

## 11. Deployment

Frontend to Vercel. Backend to Hugging Face Spaces or Render free tier. `GROQ_API_KEY` and
the Reddit credentials are set as host secrets, never committed.

**Credential note:** the Groq key currently in `.env` was pasted into a chat transcript
during design. It must be rotated at `console.groq.com` before the repository is made
public. `.env` is confirmed gitignored (`.gitignore:6`), and `.agents/` plus
`.claude/skills/` were added to `.gitignore` so installed agent skills do not ship in the
repository.

---

## 12. Open items requiring the developer

1. Create a Reddit script app at `reddit.com/prefs/apps` for the client ID and secret.
2. Accept Orpheus terms at `console.groq.com` if the neural TTS upgrade is wanted. The
   project works without this.
3. Hand-label the ~40 golden-set articles. 2–3 hours, and the single highest-value artifact
   in the repository.
4. Rotate the Groq API key before publishing.

---

## 13. What this is and is not

Delivered as designed, this is a well-engineered project: real structured LLM extraction,
a defensible evaluation methodology, a concrete rate-limit engineering story, working
tests, and honest documentation.

It is not a novel one. It does not involve new research, meaningful scale, or real users,
and the design makes no attempt to imply otherwise. For the purpose of getting interviews
for AI/ML engineer roles, engineered and honest is what the work needs to be.
