# newsninja

Turns a list of topics into a source-grounded news briefing, with audio, on
Groq's free tier. Every claim in a briefing carries a quote copied verbatim
from the article it came from, and a quote that is not in the source fails a
substring check — so hallucination is measurable here rather than asserted.

Four pieces: an analysis package, an evaluation harness that measures it, an
HTTP service over both, and a page that runs the service.

This is a portfolio project. There is no hosted demo — running it requires your
own free Groq API key.

## What it does

Given up to five topics, `newsninja`:

1. Fetches articles from Google News RSS (always available) and, if
   credentials are configured, Reddit.
2. Sends each topic's articles to a Groq model under a strict JSON schema and
   gets back structured analysis: entities, stance, a confidence score, and a
   list of claims, where each claim carries a quote copied verbatim from the
   source article.
3. Synthesises the per-topic analyses into a single spoken-word briefing
   script, and translates it if a non-English language was requested.
4. Renders the script to audio — gTTS by default, or Orpheus neural TTS as an
   opt-in upgrade for English and Saudi Arabic.

Reachable three ways: a command-line entry point, an HTTP service, and a page
that drives the service.

## How it works

```
topics
  -> sources (Google News RSS, Reddit)          newsninja/sources/
  -> per-topic structured extraction (Groq)      newsninja/analysis/extract.py
  -> briefing synthesis + translation (Groq)     newsninja/analysis/synthesize.py
  -> text-to-speech (gTTS or Orpheus)             newsninja/audio/tts.py
```

A source that fails is recorded and the run continues on what's left; a
source that's simply unavailable (no Reddit credentials) is recorded
separately and reported as skipped, not as a failure. Structured extraction
that fails schema validation is retried, with the validation error fed back
to the model, up to twice, before it raises.

The package README (`api/README.md`) covers the mechanics in more depth: why
extraction is pinned to specific models, how the token-budget limiter reads
Groq's rate-limit headers, and how quote-grounding is checked.

## Requirements

Python 3.12. macOS system Python is 3.9, which cannot run this project's
dependencies — use [`uv`](https://docs.astral.sh/uv/) to get an isolated
3.12 interpreter without installing one system-wide.

A free Groq API key from [console.groq.com/keys](https://console.groq.com/keys).

## Setup

```bash
cp .env.example .env      # then set GROQ_API_KEY in .env
cd api
uv run --python 3.12 --extra dev pytest
```

That installs dependencies into a project-local virtualenv and runs the test
suite (123 tests as of this writing, all network access mocked — no test in
this package makes a real HTTP call).

## Usage

From `api/`:

```bash
uv run --python 3.12 newsninja --topic "artificial intelligence" --topic "climate change"
```

Flags (see `api/newsninja/cli.py`):

| Flag | Meaning |
| --- | --- |
| `--topic TOPIC` | Topic to analyse. Repeat for up to 5. Required. |
| `--language LANGUAGE` | Output language code. Default `en`. |
| `--out OUT` | Path to write the audio file. Default `briefing.mp3`. |
| `--no-audio` | Print the script only; skip speech synthesis. |
| `--no-cache` | Bypass the SQLite response cache. |

Exit codes: `0` success, `1` missing or invalid configuration (no
`GROQ_API_KEY`), `2` the model never returned schema-valid output after
retries, `3` Groq's rate limit was hit (the message says roughly how long to
wait).

## Project layout

```
api/
  newsninja/
    config.py             # pydantic-settings; single source of truth
    models.py              # Article, ArticleAnalysis, Claim, Entity, Briefing
    errors.py               # typed exceptions (SourceError, ExtractionFailure, RateLimitError)
    cache.py                 # SQLite response cache, keyed on call identity
    pipeline.py               # orchestration: topics in, briefing + audio out
    cli.py                     # command-line entry point
    sources/
      base.py                   # Source protocol
      google_news.py            # RSS, no auth
      reddit.py                 # official OAuth API via praw
    analysis/
      client.py                 # Groq wrapper: strict-schema calls, retries, rate limiting
      extract.py                 # articles -> ArticleAnalysis
      synthesize.py               # analyses -> Briefing (synthesis + translation)
      grounding.py                 # quote-in-source containment check
      limiter.py                    # sliding-window token-budget limiter
      schema.py                      # Pydantic model -> strict JSON Schema
      prompts/                        # prompt text
    audio/
      tts.py                     # gTTS default, Orpheus opt-in, chunking
    api/
      app.py                    # create_app factory; error handlers, CORS, per-IP window
      routes.py                 # /health /analyze /brief /audio
      schemas.py                # wire models, kept apart from the domain models
      deps.py                   # process singletons — one client, one token budget
      ratelimit.py              # per-IP sliding window
  evals/                    # measures newsninja; newsninja never imports it
    corpus.py capture.py metrics.py golden.py bootstrap.py
    agreement.py judge.py report.py run.py
    data/corpus.jsonl         # 40 real articles, 5 topics, committed on purpose
  tests/                    # pytest, all network mocked
  Dockerfile                # the deployable image
  README.md                 # package-level documentation (read this for the design detail)

web/                        # the page that runs the service
  app/ components/ lib/ styles/tokens.css

docs/
  evals/latest.md           # the harness's report; latest.json is what the page reads
  design/specs/             # design docs
  design/plans/             # implementation plans

render.yaml                 # service blueprint
```

## Design notes

A few decisions worth reading the code for:

- **Constrained decoding, not hope.** Structured extraction uses Groq's
  strict `json_schema` response mode, which only `openai/gpt-oss-20b` and
  `openai/gpt-oss-120b` support on the free tier. `GroqClient.structured()`
  refuses any other model rather than silently falling back to prompted JSON.
- **Quote-grounding as a string check, not a model judgement.**
  `analysis/extract.py` requires every `Claim.quote` to be copied
  character-for-character from the source article. `analysis/grounding.py`
  checks that by exact substring containment — no normalisation, no fuzzy
  matching, no model in the loop. Nothing currently rejects an ungrounded
  claim at runtime; the check is a measurement utility that the evaluation
  harness (`api/evals/`) reads.
- **Tiered model routing under a shared rate limit.** Groq's free tier caps
  tokens per minute per model. Work is spread across three models —
  extraction on `gpt-oss-20b`, synthesis on `gpt-oss-120b`, translation on
  `llama-3.3-70b-versatile` — and `analysis/limiter.py` keeps a sliding
  sixty-second window per model, reserving an estimate before each call and
  correcting it against both the actual token usage and the provider's own
  `x-ratelimit-remaining-tokens` header.

## Running it as a service

```bash
cd api
uv run --python 3.12 uvicorn newsninja.api:create_app --factory --port 8000
```

Interactive schema at `http://127.0.0.1:8000/docs`.

| Endpoint | Work | Cost |
| --- | --- | --- |
| `GET /health` | liveness and the packaged version | none |
| `POST /analyze` | one topic in, one analysis out | 1 model call |
| `POST /brief` | 1–5 analyses in, one script across them | 1–2 model calls |
| `POST /audio` | a script in, audio bytes out | none |

A client makes N `/analyze` calls, then one `/brief`, then one `/audio`. The
split is arithmetic rather than taste: one topic reserves roughly 2,400 tokens
against an 8,000-per-minute ceiling, so a request taking five at once would sit
blocked in the limiter for 48 seconds and no free-tier proxy would hold it.

The whole service supports about three analyses per minute across all callers.
That is the token ceiling, not a queue — past it the service answers `429` with
a real `Retry-After` rather than holding the connection open.

The container is `api/Dockerfile`, about 55 MB, running as uid 1000 and reading
`$PORT`.

## Running the page

```bash
cd web
npm install && npm run dev        # http://localhost:3000
```

It calls the service, so start that too. Add up to five topics, pick one of
twelve languages, and it extracts each topic, writes one briefing across them,
and renders the audio. A topic refused by the rate limiter does not stop the
run — the briefing is built from whatever succeeded and the page says which
topics did not make it.

## Limitations

- **Rate limits.** Groq's free tier is tokens-per-minute limited, per model.
  Running several multi-article topics back to back can still queue on the
  limiter or, occasionally, hit a 429 (exit code `3`).
- **Orpheus TTS covers English and Saudi Arabic only**, and requires
  accepting the model's terms at console.groq.com. Until you do, every
  Orpheus call returns HTTP 400 (`model_terms_required`) and the pipeline
  silently falls back to gTTS — turning `ENABLE_ORPHEUS` on before accepting
  the terms changes nothing audible.
- **Reddit requires a free script app** from reddit.com/prefs/apps. The
  public `search.json` endpoint returns HTTP 403, so there is no
  credential-free path; without `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET`
  the Reddit source reports itself unavailable and the run proceeds on
  Google News alone.
- **Stance is a model judgement, not ground truth.** The `stance` field on
  each `ArticleAnalysis` is Groq's classification of an article's tone, with
  a confidence score attached by the same model. The evaluation harness can
  score it against human-reviewed labels (see **Project status** below), but
  no such run has happened yet.
- **Maximum 5 topics per run**, enforced by the pipeline.

## Project status

Built, tested, and measured:

| Piece | State |
| --- | --- |
| `newsninja` core package | sources, structured extraction, synthesis, translation, TTS, caching, rate limiting, CLI |
| `evals/` harness | deterministic metrics, agreement scaffolding, LLM judge, report writer |
| HTTP service | `/health` `/analyze` `/brief` `/audio`, one error envelope, per-IP window |
| Container | builds and serves; CI smoke-tests it on every push |
| `web/` site | runs the service in the browser, renders the harness's numbers |

349 tests, all offline — no test makes a network call. `ruff` clean.
`mypy --strict` clean across 37 files with zero `type: ignore` and exactly one
`# noqa`, on the intentional Orpheus fallback. CI runs three jobs: the Python
suite, a container build with a smoke test, and a site build that fails if the
page stops publishing the eval numbers.

### What has actually been measured

`docs/evals/latest.md` is committed, and the page renders from its JSON sibling
rather than keeping a copy. The deterministic metrics are real arithmetic over
real output.

**The agreement metrics are not measured.** `evals/data/golden.jsonl` holds five
drafted labels, none of them reviewed, and a validator refuses to let a
`model-drafted` label be marked `reviewed` — model output cannot become its own
ground truth. Until a human corrects them, the agreement section reads
`unavailable` and this README cites no accuracy figure.

One measurement worth reading with its denominator: a run reported a quote
grounding rate of `1.00` over three claims, and a later run `0.81` over
twenty-one. Both are real arithmetic; only the second says much. The report
prints the claim count directly above the rate for that reason, and the reason
the counts are low is upstream — Google News RSS carries no article prose, so
the median article body across the corpus is 195 characters, roughly a headline.
That bounds what "grounded" can mean here, and the report says so on its face.

### Not done

- **No deployment.** `render.yaml` describes the service and `web/README.md`
  describes the site; neither has been pointed at an account.
- **Reddit is unconfigured.** The source is written and tested; it needs a free
  script app. Its `selftext` is the one plausible route to prose worth quoting.
- **No frontend test suite.** The page renders what the service returns and what
  the harness wrote; the behaviour worth testing lives in the Python package.

## License

MIT. See [LICENSE](LICENSE).
