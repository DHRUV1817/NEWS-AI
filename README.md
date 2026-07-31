# newsninja

Turns a list of topics into a source-grounded news briefing, with audio, on
Groq's free tier. The core analysis package is built and tested; the web
service and frontend described below are planned, not shipped yet.

This is a portfolio project. There is no live demo URL — running it requires
your own free Groq API key.

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

Everything runs from the command line; there is no server process.

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
  tests/                    # pytest, all network mocked
  README.md                 # package-level documentation (read this for the design detail)

docs/
  superpowers/specs/         # design docs
  superpowers/plans/          # implementation plans
```

`api/evals/` (evaluation harness) exists and is covered below. `api/newsninja/api.py`
(FastAPI service) and `web/` (Next.js frontend) are referenced in the design docs
under `docs/superpowers/specs/` but do not exist in the repository yet.

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

Built and tested: the `newsninja` core package — sources, structured
extraction, synthesis, translation, TTS, caching, rate limiting, and the CLI.

Also built and tested: the evaluation harness (`api/evals/`), covered in
`api/README.md`. It computes three families of metrics — deterministic
(schema validity, quote-grounding, ungrounded-claim rate), agreement against a
human-reviewed golden set (entity precision/recall/F1, stance accuracy and
Cohen's kappa), and LLM-judged rubric scores — and `python -m evals.run
--report` writes them to `docs/evals/latest.md`. The harness has not yet been
run against a reviewed golden set: `evals/data/golden.jsonl` has not been
created yet (no label anywhere has `reviewed: true`), and `docs/evals/latest.md`
does not exist. So this README makes no claims about accuracy, performance, or
scale — there is nothing measured yet to cite.

Not yet built: the FastAPI service and the Next.js frontend, designed in
`docs/superpowers/specs/2026-07-31-newsninja-portfolio-design.md` but not
implemented.

## License

MIT. See [LICENSE](LICENSE).
