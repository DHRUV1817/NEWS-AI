# NewsNinja — FastAPI Service Design (Plan 3)

**Date:** 2026-07-31
**Status:** Awaiting review
**Depends on:** Plan 1 (core analysis package, merged), Plan 2 (evaluation harness, pushed)

---

## 1. What this is

An HTTP layer over the analysis package, so the Next.js frontend of Plan 4 has something
to call and so the project can be demonstrated without a terminal.

The organising principle from the portfolio design carries through unchanged: **every
claim the service makes must be mechanically checkable.** A service that appears to work
and quietly returns degraded results is worse than one that says what went wrong.

---

## 2. The measurement this design rests on

The shape of this service is not a matter of taste. It follows from one number.

`openai/gpt-oss-20b` has a free-tier ceiling of **8,000 tokens per minute**. Measured
against the committed corpus (`api/evals/data/corpus.jsonl`, 40 real articles across 5
topics), a single-topic extraction reserves:

| Topic | Reserved tokens |
| --- | --- |
| artificial intelligence | 2,461 |
| climate change | 2,295 |
| cryptocurrency | 2,340 |
| space exploration | 2,385 |
| renewable energy | 2,306 |
| **Total for five topics** | **11,787** |

Reservation is `_estimate_tokens(system + user) + EXPECTED_COMPLETION_TOKENS`
(`analysis/client.py:191`), which is what `TokenBudgetLimiter` actually books.

Simulating `TokenBudgetLimiter` with an injected clock, five topics in sequence at an
assumed 4s of API latency per call:

| Topic | Limiter sleep | Elapsed after |
| --- | --- | --- |
| artificial intelligence | 0.0s | 4.0s |
| climate change | 0.0s | 8.0s |
| cryptocurrency | 0.0s | 12.0s |
| space exploration | **48.0s** | 64.0s |
| renewable energy | 0.0s | 68.0s |

The fourth topic does not fit the 60-second window, so `reserve()` blocks in `time.sleep`
(`analysis/limiter.py:76`) for 48 seconds. **The extraction phase alone takes 68 seconds**
before source fetching, synthesis, translation, or speech.

Honest about the inputs: the 4s per-call latency is an assumption, and `settle()` may
return budget if completions come in under the 1,500-token estimate. The 48-second sleep
is not an assumption — it follows from the reservation sizes and the window length.

### What that rules out

`MAX_TOPICS = 5` and "one synchronous request" are incompatible on free hosting. Any
endpoint accepting all five topics produces a request that a platform proxy will sever
(~100s on Render) or that a browser will abandon.

**Deploy target is deliberately undecided** between Hugging Face Spaces and Render, so
this design targets the stricter of the two: **no request may exceed ~100s, with 65s as
the working budget** so a cold start still fits.

---

## 3. Approach: expose the seams that already exist

`extract_topic` and `build_briefing` are already separate public functions; `run_pipeline`
is only their composition. Splitting the API along that existing joint is not a rewrite.

| Endpoint | Calls | Cost |
| --- | --- | --- |
| `POST /analyze` | `extract_topic`, one topic | 1 LLM call, ~2.4k tokens |
| `POST /brief` | `build_briefing` over supplied analyses | 1–2 LLM calls |
| `POST /audio` | `default_tts` on a supplied script | no LLM call |
| `GET /health` | — | — |

The client makes N `/analyze` calls, then one `/brief`, then one `/audio`. Every request
is short by construction — no batch caps, no partial-batch semantics — and the unified
briefing survives, because `/brief` still sees every analysis at once.

This also serves the locked frontend design (portfolio spec §8): macrostructure **05
Workbench** wants the tool visibly working. Topics resolving one at a time is that. A
68-second spinner is not.

### Alternatives rejected

**Batch the topics, cap the batch at 3.** Keeps `run_pipeline` intact, but
`build_briefing` synthesises *across* analyses into one script. Five topics in two batches
yields two disconnected briefings rather than one. The unified script is the product; a
batching artifact is not a good enough reason to lose it.

**Job-and-poll (`POST /jobs` → `202`, `GET /jobs/{id}`).** Handles any duration and keeps
one briefing, but needs a job store, a background worker, and lifecycle tests — and free
hosting sleeps or recycles instances, so a 90-second job can die with the container. It
buys durability the platform will not honour, at the highest complexity cost of the three.

**Cost accepted:** `run_pipeline` becomes unused *by the API*. The CLI and the eval
harness continue to use it, so it stays covered and stays the thing under measurement.

---

## 4. Structure

The portfolio spec reserves `api/newsninja/api.py` as a single file. Four endpoints plus
schemas plus rate limiting plus CORS lands near 350 lines there, past the point where the
file does one thing. **This design deliberately diverges** and makes it a package,
matching `analysis/` and `audio/`:

```
api/newsninja/api/
  __init__.py     # exports `app`
  app.py          # FastAPI app, routes, exception handlers
  schemas.py      # request/response models, separate from domain models
  ratelimit.py    # per-IP window
  deps.py         # shared client, cache, sources — built once, not per request
```

### `deps.py` carries an invariant

`GroqClient` owns the `TokenBudgetLimiter`. If each request constructs a new client, every
request gets a **fresh, empty budget window**; the limiter never waits, and the service
walks directly into Groq 429s. The client must be one long-lived instance per process.

This failure is silent — nothing raises, throughput just quietly exceeds the ceiling — so
it gets an explicit test (§8).

---

## 5. Changes to existing code

### 5.1 Factor one topic out of `run_pipeline`

`/analyze` needs fetch-then-extract for a single topic. That loop exists at
`pipeline.py:96-107`. Rather than duplicate it:

```python
def analyze_topic(topic, sources, client, cache=None, limit=8) -> TopicResult
    # -> (analysis, source_errors, skipped_sources)
```

`run_pipeline` then calls it in its loop. **No behaviour change**: same fetch order, same
per-source error collection, same `_empty()` fallback when no articles are found.

This is surgery on code already merged to `main` and covered by passing tests. The
regression bar is stated in §8: the existing suite must stay green *untouched*.

### 5.2 Let the limiter refuse instead of only waiting

```python
def reserve(self, estimated_tokens: int, max_wait: float | None = None) -> int
```

If the required wait exceeds `max_wait`, raise `RateLimitError` carrying the declined wait
as `retry_after`, instead of sleeping. `max_wait=None` is the default and preserves
current behaviour exactly, so the CLI and eval harness are untouched.

`GroqClient` accepts the same ceiling at construction and passes it through. The API
supplies one; nothing else does.

**Two code paths sleep, not one.** The window loop (`limiter.py:76`) and the forced-wait
branch (`limiter.py:70`) fed by `observe()`. Both must respect the ceiling — bounding only
the first lets a provider-signalled backoff hold the socket open anyway.

Waiting is correct for a CLI: waiting beats failing. It is wrong inside an HTTP handler,
where it converts a rate-limit condition into a held-open connection that a proxy
eventually severs and a visitor reads as a frozen page.

---

## 6. Endpoint contracts

### `POST /analyze`

```json
{ "topic": "artificial intelligence", "limit": 8 }
```

`topic`: 1–200 characters, non-blank. `limit`: 1–12, default 8.

Response — `analysis` is an `ArticleAnalysis` serialised unchanged:

```json
{
  "analysis": {
    "topic": "artificial intelligence",
    "summary": "…",
    "entities": [{ "name": "OpenAI", "kind": "org" }],
    "stance": "neutral",
    "confidence": 0.72,
    "key_claims": [{ "text": "…", "quote": "…" }]
  },
  "source_errors":   { "reddit": ["reddit: 403 Forbidden"] },
  "skipped_sources": ["reddit"]
}
```

`source_errors` and `skipped_sources` remain **separate fields**, per the invariant in the
handoff §6. A skipped source is a missing credential the visitor could supply; a failed
source is one that was tried and broke. They are different facts and must not collapse
into a single "problems" list.

Returns `200` even when both are populated. A degraded briefing is a success — that is
already how `run_pipeline` behaves, and the HTTP layer must not disagree with the package
it wraps.

### `POST /brief`

```json
{ "analyses": [ { "topic": "…", "summary": "…", "…": "…" } ], "language": "en" }
```

`analyses` holds 1 to 5 `ArticleAnalysis` objects in the same shape `/analyze` returns.

`language` is validated against `^[a-z]{2}(-[A-Za-z]{2})?$`. It reaches a translation
prompt, so it is not a free-text field. The 1–5 bound imports `MAX_TOPICS` rather than
restating the number.

Returns the `Briefing` unchanged: `topics`, `script`, `analyses`, `language`.

**Stated limitation.** `/brief` renders whatever analyses it receives. The server holds no
articles at that point and cannot re-check that quotes are verbatim. The grounding
guarantee belongs to `/analyze`, which produced them. A caller can POST invented analyses
and receive a script.

This is acceptable — it is the caller's own data, and the eval harness measures the
pipeline rather than the HTTP layer. It is written down because a reader would otherwise
reasonably assume `/brief` validates, and this project does not let readers assume.

### `POST /audio`

```json
{ "script": "…", "language": "en" }
```

`script` capped at **20,000 characters**. Not cosmetic: `chunk_text` splits at 3,000
characters and gTTS makes one outbound call per chunk, so an uncapped script is an
amplification vector — 1 MB of text becomes roughly 350 requests from the host.

Returns raw bytes. `Content-Type` is `audio/mpeg` for gTTS or `audio/wav` when Orpheus is
enabled, since `synthesize_speech` returns different formats. Orpheus terms are unaccepted
(handoff §5), so in practice this is `audio/mpeg` today.

The speech callable comes from `default_tts(settings.groq_api_key)` — the key is required
because Orpheus is a separate REST endpoint, not part of the chat client
(`pipeline.py:52-72`). It is built once in `deps.py`, not per request.

### `GET /health`

```json
{ "status": "ok", "version": "3.0.0" }
```

`version` is read from package metadata via `importlib.metadata.version("newsninja")`, not
written as a literal — a hardcoded string drifts from `pyproject.toml` silently, and a
version that lies is worse than no version field.

Liveness only. It deliberately does not call Groq: a health check that spends tokens
against an 8,000 TPM budget is a health check that causes outages.

### Error envelope

```json
{ "error": { "type": "rate_limit", "message": "…", "retry_after": 48.0 } }
```

| Raised | Status | Extra |
| --- | --- | --- |
| `ValueError` | 422 | — |
| request schema invalid (`RequestValidationError`) | 422 | same envelope, plus `detail`: `exc.errors()` |
| `RateLimitError` | 429 | `Retry-After` header **and** `retry_after` in body |
| `SourceError` | 502 | names the source |
| `ExtractionFailure` | 502 | — |

`Retry-After` appears in both header and body. The header is what proxies and browsers
honour; the body is what the frontend can render without reading headers through CORS.

`RequestValidationError` does not subclass `ValueError` and FastAPI pre-registers its own
handler for it, so it needs its own explicit registration or a bad request body answers in
FastAPI's `{"detail": [...]}` shape instead of this service's one envelope. The validation
detail is preserved rather than discarded, under an optional `detail` key, run through
`jsonable_encoder` first — `exc.errors()` can carry a raised exception in a field validator's
`ctx`, which is not JSON-serialisable on its own.

---

## 7. Hardening

### Two defences, two different jobs

Per-IP limiting does not protect the token budget, and conflating them leaves a hole.

**Per-IP window** (`ratelimit.py`) stops one client hammering the service. In-process
sliding window, default 10 requests per minute per IP.

**Bounded reserve** (§5.2) protects the shared token budget. At ~2,400 tokens per analysis
against 8,000 TPM, the service supports roughly **three analyses per minute across all
visitors combined**. Ten well-behaved callers from ten IPs each pass the per-IP check and
still exhaust the budget. Only the bounded reserve catches that, and it answers `429` with
a true `Retry-After` rather than hanging.

**Limitations, stated rather than discovered later:** the per-IP window is in-process, so
it resets on restart and does not hold across multiple instances. On a free single-instance
host that is acceptable.

Behind a proxy, `request.client.host` is the proxy's address and every visitor shares one
bucket. Reading `X-Forwarded-For` fixes that but is spoofable unless the platform
overwrites the header. Hence `trust_proxy_headers: bool = False`, off by default; enabling
it trusts the platform to sanitise.

### CORS

```python
allow_origins=settings.allowed_origins   # [] by default, never "*"
allow_credentials=False                  # no cookies are used
expose_headers=["Retry-After"]
```

`expose_headers` is load-bearing. CORS hides all but a six-header safelist from browser
JavaScript, and `Retry-After` is not on it. Without this line the frontend receives the
`429` but cannot read how long to wait — visible in DevTools, invisible to `fetch`.

### New `Settings` fields

| Field | Default | Purpose |
| --- | --- | --- |
| `allowed_origins` | `[]` | the Vercel origin |
| `api_max_wait_seconds` | `5.0` | bounded-reserve ceiling |
| `rate_limit_per_minute` | `10` | per-IP window |
| `trust_proxy_headers` | `False` | read `X-Forwarded-For` |

New dependencies: `fastapi>=0.115`, `uvicorn[standard]>=0.34`. `httpx` is already a
dependency and is what `TestClient` requires, so no new dev dependency.

---

## 8. Test plan

Tests are written first. No test may touch the network.

**Unit**

- Bounded reserve: sleeps when the wait is under the ceiling; raises `RateLimitError`
  carrying the declined wait when over; **the `observe()` forced-wait path is bounded
  too**; `max_wait=None` behaves exactly as today.
- Per-IP window: allows N, refuses N+1, recovers after the window, keeps IPs independent,
  honours `trust_proxy_headers` in both settings.

**API — `TestClient` against a stub client**

- Each endpoint's happy path and response shape.
- `source_errors` and `skipped_sources` stay distinct; a degraded run still returns `200`.
- Each typed error maps to its status; `429` carries `Retry-After` in header *and* body.
- `/audio` returns the correct `Content-Type`; a 20,001-character script is `422`.
- Six analyses to `/brief` is `422`.
- CORS preflight allowed and refused; `Retry-After` readable by the browser.
- **The shared client is one instance across requests** (§4). Silent when broken, so it is
  asserted directly.

**Regression**

The existing suite must stay green through the `analyze_topic` refactor **without
edits**. A test that needs changing means the refactor altered behaviour and is wrong.

Fixes get mutation-tested, per the process note in the handoff §11: six mutations
previously survived a suite that looked green, so "a test exists" is not evidence. Mutation
backups are written to `/tmp`, never into the repository.

---

## 9. Out of scope

**Actual deployment.** It needs an account, a rotated Groq key (handoff §9.1), and choices
only the owner can make. This plan produces a service that runs locally under
`uvicorn newsninja.api:create_app --factory` and is *ready* to deploy. It does not
deploy it.

The entrypoint is the factory rather than a module-level `app` deliberately. Building
the application at import time constructs `Settings`, which reads the credentials file
on *any* import of the package — so a machine without a configured key cannot even
import the module to run an unrelated test. Deferring construction to the server means
credentials are read when a server starts and at no other time.

**Authentication.** The service is public and unauthenticated by design; the bounded
reserve and per-IP window are what stand between it and a drained key.

**Frontend.** Plan 4, designed separately and already locked.

---

## 10. Success criteria

1. Every endpoint's worst case is bounded by construction, not by hope: at most
   `api_max_wait_seconds` (5s) of limiter wait plus the calls listed in §3 — one LLM call
   for `/analyze`, two for `/brief`, none for `/audio`. Past that ceiling the service
   returns `429` instead of waiting. Demonstrated by the bounded-reserve tests, not by
   timing a live run.
2. The five error conditions map to the five documented statuses, each with a test.
3. `ruff` clean, `mypy --strict` clean, zero `type: ignore`, and no new `# noqa` beyond the
   single existing one in `audio/tts.py`.
4. The existing test suite passes unmodified.
5. A reader of the README can verify every claim it makes about the service by running a
   command.
