# newsninja (core package)

Turns a list of topics into a source-grounded news briefing with audio, on
Groq's free tier.

## Requirements

Python 3.12. The system Python on macOS is 3.9 and will not work — use `uv`.

## Setup

    cp ../.env.example ../.env    # then set GROQ_API_KEY
    uv run --python 3.12 --extra dev pytest

## Usage

    uv run --python 3.12 newsninja --topic "artificial intelligence" --topic "climate"

Exit codes: `0` success, `1` missing or invalid configuration, `2` the model
never returned schema-valid output, `3` the provider's rate limit was hit — the
message says roughly how long to wait.

## Design notes

**Structured output.** Extraction uses Groq's strict `json_schema` mode, which is
supported only on `openai/gpt-oss-20b` and `openai/gpt-oss-120b`; `GroqClient`
refuses any other model rather than quietly falling back to free-form text.
Pydantic's schema output needs a transform first — see `analysis/schema.py`. A
response that fails validation is retried with the validation error fed back to
the model, and raises `ExtractionFailure` once the retry budget is spent.

**Grounding.** The extraction prompt requires every `Claim` to carry a `quote`
copied character-for-character from the supplied article text.
`analysis/grounding.py` is what checks it: `grounding_rate` reports the fraction
of quotes appearing verbatim in the concatenated sources, by exact substring
containment — no normalisation, no fuzzy matching, no model in the loop. That is
the sense in which hallucination here is a string containment check rather than
a judgement.

What it does not mean: nothing rejects an ungrounded claim at runtime. The check
is a measurement utility, deliberately outside the pipeline's control flow, and
is what the eval harness will read.

**Rate limits.** The free tier is tokens-per-minute limited, per model, so work is
spread across models — extraction on `gpt-oss-20b`, synthesis on `gpt-oss-120b`,
translation on `llama-3.3-70b-versatile` — and articles are batched one call per
topic rather than one per article. `analysis/limiter.py` holds a sliding
sixty-second window per model: a call reserves an estimate that includes the
completion tokens it expects to spend, then settles that reservation against the
usage the provider actually reported, and the provider's own
`x-ratelimit-remaining-tokens` header forces a wait when it disagrees with the
local view. A model with no configured budget raises rather than skipping the
limiter.

**Audio.** gTTS by default, across twelve languages, returning MP3. Orpheus
(`ENABLE_ORPHEUS=true`) is an opt-in upgrade covering English and Saudi Arabic
only; it POSTs to Groq's `/openai/v1/audio/speech` endpoint and returns WAV.
`--out` is written verbatim, so name the file for whichever backend you expect.

Orpheus returns HTTP 400 `model_terms_required` until its terms are accepted at
console.groq.com, so on a fresh account every Orpheus call fails and gTTS — the
fallback — is the path that actually runs. Long scripts are chunked, and no
chunk exceeds the limit even when the text carries no punctuation the splitter
recognises.

**Sources.** Google News RSS needs no authentication. Reddit requires a free
script app at reddit.com/prefs/apps — the public JSON endpoint returns 403 — and
without credentials the source reports itself unavailable, which the CLI prints
as a skipped source, distinct from a failure. A source that is tried and fails
has every failure recorded against its name and the run continues on what is
left.

## Evaluation

`api/evals/` measures the extraction pipeline, split into three families that
are reported separately because they carry different epistemic weight:

- **Deterministic** (`evals/metrics.py`) — schema validity, quote-grounding
  rate, and ungrounded-claim rate. Pure arithmetic over the model's output and
  its source articles; no model judges any of it, and a metric that cannot be
  computed is `None`, never `0.0`.
- **Agreement** (`evals/agreement.py`) — entity precision/recall/F1 and stance
  accuracy plus Cohen's kappa, scored against the golden set. Kappa is
  reported alongside accuracy because it discounts the agreement you would get
  by chance.
- **Judged** (`evals/judge.py`) — a rubric score (coverage, neutrality,
  coherence, 1-5) from one Groq model grading another model's summary. This is
  the least objective family and is presented as one model's opinion, not
  ground truth.

Deterministic and agreement count entities differently, on purpose. Agreement
case-folds
each topic's entities into a set before scoring, so a topic naming "Apple"
four times contributes one entity and repetition cannot buy precision.
`mean_entities_per_topic` counts every emission, so the same topic contributes
four — it describes how much the model said, not how much of it was distinct.
Expect the two numbers to disagree on identical data; neither is a typo for
the other.

The golden set (`evals/data/golden.jsonl`) is meant to be model-drafted and
human-corrected: `python -m evals.bootstrap` drafts candidate labels with
`gpt-oss-120b` for a human to review and correct, but every drafted label
starts `reviewed: false` and is invisible to the agreement metrics until a
human sets `reviewed: true`. Presenting an unreviewed, model-drafted label as
ground truth would make the agreement numbers circular.

No golden set has been created yet and no run has happened, so this README
cites no metric value from any of the three families — there is nothing
measured yet to cite. Until `evals/data/golden.jsonl` exists with reviewed
labels, the agreement section of a generated report reads `unavailable`.

Commands, run from `api/`:

```bash
python -m evals.capture           # fetch and commit a stable article corpus
python -m evals.bootstrap         # draft golden labels for human correction
python -m evals.run --report      # run all three metric families, write docs/evals/latest.md
```

All three are human-invoked and not part of CI; `evals.run` makes real Groq
API calls against the free tier and is rate-limited by the same limiter
production uses.

`evals.run` validates the corpus and the golden set before it builds a client
or spends a token: duplicate topics in either abort the run with exit code 2
and a message naming them, having cost nothing. Duplicate labels a human has
not reviewed only warn — they back no reported number. Topics whose capture
found no articles are skipped rather than extracted, and the report's header
says how many were skipped so `Topics evaluated` can be reconciled against the
corpus.

## HTTP service

```bash
uv run --python 3.12 uvicorn newsninja.api:create_app --factory --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

The endpoints are split so no single request runs long. That is arithmetic, not
preference: `openai/gpt-oss-20b` allows 8,000 tokens per minute, one topic
reserves roughly 2,400, and five topics reserve 11,787 — so a five-topic request
sits in the token limiter for 48 seconds and no free-tier proxy will hold the
connection. Measured against `evals/data/corpus.jsonl`; see
`docs/superpowers/specs/2026-07-31-fastapi-service-design.md`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness and package version. Never calls the model. |
| `POST /analyze` | One topic in, one `ArticleAnalysis` out. |
| `POST /brief` | 1–5 analyses in, one unified script out. |
| `POST /audio` | A script in, audio bytes out. |

A client makes N `/analyze` calls, then one `/brief`, then one `/audio`. Feed
`/brief` the `analysis` object out of each `/analyze` response, not the response
itself. `/brief` answers `{"briefing": {...}}`.

Two bounds a caller meets as a `422`: `/brief` accepts at most 9,000 characters
of analysis text across the whole request, and both `/brief` and `/audio` accept
only the twelve language codes the speech layer supports — a code outside that
set would be silently spoken in English, so it is refused instead.

### Failures

| Condition | Status |
| --- | --- |
| invalid request body | 422 |
| token budget exhausted | 429, with `Retry-After` |
| a source failed | 200, reported in `source_errors`; the briefing is degraded, not failed |
| the model would not produce valid output | 502 |

A source failing does not fail the request: `/analyze` catches `SourceError`
per source and still returns its `ArticleAnalysis`, with the broken sources
named in `source_errors` — even when every source fails, the response is a 200
with an empty analysis rather than an error. A `502` handler for `SourceError`
is still registered at the application level, but nothing on the `/analyze`
path can currently raise one past that per-source catch, so a caller of this
API will not see it.

Every failure uses one envelope, including an invalid request body:

```json
{ "error": { "type": "rate_limit", "message": "…", "retry_after": 48.0 } }
```

An invalid request body carries the same envelope, with pydantic's per-field errors
under `detail` instead of `retry_after`:

```json
{ "error": { "type": "invalid_request", "message": "…", "detail": [ ... ] } }
```

### What this does not do

`/brief` renders whatever analyses it is given. The server holds no articles at
that point and cannot check that quotes are verbatim, so the grounding guarantee
belongs to `/analyze`, which produced them.

The per-IP window is in-process: it resets on restart and does not hold across
multiple instances. It bounds one client, not the shared token budget — the
limiter's wait ceiling does that.

`TRUST_PROXY_HEADERS` is off by default. Turning it on trusts the platform to
overwrite `X-Forwarded-For`; on a platform that does not, the per-IP window
becomes bypassable by setting the header.

The response cache is an unbounded public write surface. `/analyze` writes one
row per distinct topic, `Cache` has no TTL and no size cap, and nothing in the
service calls `clear()` — so the SQLite file grows with the number of distinct
topics anyone has ever asked for, and only deleting `CACHE_PATH` shrinks it.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | required | — |
| `ALLOWED_ORIGINS` | `[]` | JSON list of browser origins |
| `API_MAX_WAIT_SECONDS` | `5.0` | limiter wait before answering 429 |
| `RATE_LIMIT_PER_MINUTE` | `10` | per-IP request ceiling; at least 1 |
| `TRUST_PROXY_HEADERS` | `false` | read `X-Forwarded-For` |
| `ENABLE_ORPHEUS` | `false` | ask for Orpheus speech instead of gTTS |

`ENABLE_ORPHEUS` is the only variable that can change a response `Content-Type`.
It changes what is *asked for*, not what comes back: every Orpheus failure falls
back to gTTS, so `/audio` reports the media type of the bytes it actually
produced — `audio/wav` only for a successful Orpheus call, `audio/mpeg`
otherwise, including for the fallback. With the terms unaccepted that fallback
is the live path.

## Not here yet

No web UI. `api/evals/` (above) covers extraction quality; there is no
evaluation of synthesis, translation, or audio output yet. No test in this
package makes a network call.
