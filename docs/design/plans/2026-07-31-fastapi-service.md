# FastAPI Service Implementation Plan (Plan 3)

> Execute this plan task by task, reviewing each task before starting the next.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put an HTTP layer over the existing analysis package so the Plan 4 frontend has something to call, with no request long enough for a free-tier proxy to sever.

**Architecture:** The API splits along seams the package already has. `extract_topic` and `build_briefing` are separate functions today and `run_pipeline` is only their composition, so `POST /analyze` handles one topic, `POST /brief` synthesises across all analyses, and `POST /audio` renders a supplied script. The client orchestrates. Two supporting changes make this safe: `run_pipeline`'s per-topic loop is lifted into a reusable `analyze_topic`, and `TokenBudgetLimiter` learns to refuse with a `429` instead of only sleeping.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, pydantic v2, pydantic-settings, pytest. All commands run through `uv`.

**Spec:** `docs/design/specs/2026-07-31-fastapi-service-design.md`

## Global Constraints

- **Every command runs through `uv`.** System Python is 3.9.6 and cannot import this project's dependencies. Use `uv run --python 3.12 --extra dev <cmd>` from the `api/` directory. Never invoke bare `python` or `pytest`.
- **No third-party attribution** in commit messages, code comments, or documentation — name no tool, vendor, or assistant, only the author. No `Co-Authored-By` trailers, no "Generated with" footers. This binds the documents themselves: they are committed to a public repository.
- **Commit subjects are descriptive sentences**, matching existing history ("Add eval runner, markdown report, and documentation"). Do **not** use `feat:` / `fix:` conventional-commit prefixes — this repo does not use them.
- **`mypy --strict` must stay clean with zero `type: ignore`.** Verify with `uv run --python 3.12 --extra dev mypy newsninja evals`.
- **Exactly one `# noqa` may exist in the tree** — `BLE001, S110` in `newsninja/audio/tts.py:231`. Do not add another.
- **`ruff check .` must stay clean.**
- **No test may touch the network.** `tests/conftest.py` already scrubs credentials from the environment for every test.
- **The existing 223 tests must pass unmodified.** If a task requires editing an existing test, the change altered behaviour and is wrong — stop and report rather than editing the test.
- **`newsninja` must never import `evals`.** The dependency runs one way only.
- **Run `git check-ignore -v <path>` before assuming any new file will be tracked.** The `.gitignore` is written around broad category excludes and has swallowed four legitimate source paths so far.
- **Commit after every task.** Long-running work has been interrupted mid-task before; incremental commits mean an interruption cannot discard work.

---

### Task 1: Teach the limiter to refuse instead of only waiting

`TokenBudgetLimiter.reserve` currently has one response to a full budget: sleep. That is correct for the CLI, where waiting beats failing. Inside an HTTP handler it converts a rate-limit condition into a held-open socket that a proxy eventually severs and a visitor reads as a frozen page.

**Files:**
- Modify: `api/newsninja/analysis/limiter.py`
- Test: `api/tests/test_limiter.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `TokenBudgetLimiter.reserve(estimated_tokens: int, max_wait: float | None = None) -> int`. Raises `newsninja.errors.RateLimitError` when the required wait exceeds `max_wait`. `max_wait=None` preserves today's behaviour exactly.

- [ ] **Step 1: Write the failing tests**

Append to `api/tests/test_limiter.py`. The file already defines `FakeClock` and `_limiter` — reuse them, do not redefine them.

```python
from newsninja.errors import RateLimitError


def test_bounded_reserve_refuses_when_the_wait_exceeds_the_ceiling():
    limiter, clock = _limiter(tpm=1000)
    limiter.reserve(800)
    with pytest.raises(RateLimitError):
        limiter.reserve(400, max_wait=5.0)
    assert clock.slept == [], "a refused reservation must not sleep at all"


def test_bounded_reserve_reports_the_declined_wait_as_retry_after():
    """retry_after must be the wait the limiter declined, not the ceiling.

    Reporting the ceiling would tell the caller to come back in 5s when the
    budget needs 60s, producing a retry storm against a budget already full.
    """
    limiter, clock = _limiter(tpm=1000)
    limiter.reserve(800)
    clock.now += 10.0
    with pytest.raises(RateLimitError) as caught:
        limiter.reserve(400, max_wait=5.0)
    assert caught.value.retry_after == pytest.approx(50.0)


def test_bounded_reserve_sleeps_when_the_wait_fits_the_ceiling():
    limiter, clock = _limiter(tpm=1000)
    limiter.reserve(800)
    clock.now += 55.0
    limiter.reserve(400, max_wait=30.0)
    assert clock.slept == [pytest.approx(5.0)]


def test_bounded_reserve_also_bounds_the_forced_wait_path():
    """observe() sets a forced wait on a different code path from the window loop.

    Bounding only the window loop lets a provider-signalled backoff hold the
    socket open anyway, which is the exact failure the ceiling exists to prevent.
    """
    limiter, clock = _limiter(tpm=1000)
    limiter.observe(remaining=10, reset_seconds=90.0)
    with pytest.raises(RateLimitError) as caught:
        limiter.reserve(100, max_wait=5.0)
    assert caught.value.retry_after == pytest.approx(90.0)
    assert clock.slept == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd api
uv run --python 3.12 --extra dev pytest tests/test_limiter.py -k bounded -v
```

Expected: 4 failures. The first three fail with `TypeError: reserve() got an unexpected keyword argument 'max_wait'`. The fourth fails the same way.

- [ ] **Step 3: Implement the bounded reserve**

In `api/newsninja/analysis/limiter.py`, add the import at the top:

```python
from newsninja.errors import RateLimitError
```

`errors.py` imports nothing from this package, so this introduces no cycle.

Add this private helper to the class, above `reserve`:

```python
    def _wait_or_refuse(self, wait: float, max_wait: float | None) -> None:
        """Sleep for ``wait``, unless a ceiling says the caller cannot afford it.

        ``retry_after`` carries the wait that was declined rather than the
        ceiling: the caller needs to know when the budget actually frees, not
        how long this particular caller was willing to hold on.
        """
        if max_wait is not None and wait > max_wait:
            raise RateLimitError(
                f"the token budget needs {wait:.1f}s to clear, which exceeds "
                f"this caller's {max_wait:.1f}s ceiling",
                retry_after=wait,
            )
        self._sleep(wait)
```

Then replace the body of `reserve` so both sleeping paths route through it:

```python
    def reserve(self, estimated_tokens: int, max_wait: float | None = None) -> int:
        """Block until ``estimated_tokens`` fits inside the budget, then book it.

        With ``max_wait`` set, a wait longer than the ceiling raises
        ``RateLimitError`` instead of sleeping. Callers that can afford to wait
        — the CLI, the eval harness — pass nothing and behave as before.

        Returns a reservation id to hand to ``settle`` once the real cost of the
        call is known.
        """
        if estimated_tokens > self._tpm:
            raise ValueError(
                f"reservation of {estimated_tokens} exceeds the entire "
                f"per-minute budget of {self._tpm}; split the request"
            )

        now = self._clock()
        if now < self._forced_wait_until:
            self._wait_or_refuse(self._forced_wait_until - now, max_wait)
            now = self._clock()

        while self._used(now) + estimated_tokens > self._tpm:
            oldest_at, _, _ = self._events[0]
            self._wait_or_refuse(max(0.0, oldest_at + WINDOW_SECONDS - now), max_wait)
            now = self._clock()

        reservation = self._next_reservation
        self._next_reservation += 1
        self._events.append((now, estimated_tokens, reservation))
        return reservation
```

- [ ] **Step 4: Run the full limiter suite**

```bash
uv run --python 3.12 --extra dev pytest tests/test_limiter.py -v
```

Expected: all pass, including the pre-existing tests. Those pre-existing tests are the proof that `max_wait=None` is unchanged — **do not edit them**.

- [ ] **Step 5: Run the whole suite and the type checker**

```bash
uv run --python 3.12 --extra dev pytest -q
uv run --python 3.12 --extra dev mypy newsninja evals
uv run --python 3.12 --extra dev ruff check .
```

Expected: 227 passed, mypy `Success`, ruff `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add api/newsninja/analysis/limiter.py api/tests/test_limiter.py
git commit -m "Let the token limiter refuse a wait it cannot afford

Sleeping is the right answer in a CLI, where waiting beats failing. In a
request handler it turns a rate limit into a socket held open until the
platform proxy severs it, which a visitor reads as a frozen page.

reserve() now takes an optional ceiling and raises RateLimitError carrying
the wait it declined. Both sleeping paths are bounded, not just the window
loop: observe() sets a forced wait on a separate branch, and leaving that
one unbounded would let a provider-signalled backoff hang the request anyway.

Passing no ceiling behaves exactly as before, so the CLI and the eval
harness are untouched."
```

---

### Task 2: Give `GroqClient` a ceiling to pass through

The limiter can now refuse, but nothing asks it to. `GroqClient` owns the limiters, so the ceiling has to travel through it.

**Files:**
- Modify: `api/newsninja/analysis/client.py:161-192`
- Test: `api/tests/test_client.py`

**Interfaces:**
- Consumes: `TokenBudgetLimiter.reserve(estimated_tokens, max_wait=None)` from Task 1.
- Produces: `GroqClient(api_key: str, limiters: dict[str, TokenBudgetLimiter] | None = None, transport: Transport | None = None, max_wait: float | None = None)`.

- [ ] **Step 1: Write the failing test**

Append to `api/tests/test_client.py`. `StubTransport` and `FakeClock` already exist in that file — reuse them.

```python
def test_a_client_with_a_ceiling_refuses_rather_than_waiting():
    """The ceiling has to reach the limiter, not merely be stored on the client."""
    clock = FakeClock()
    limiter = TokenBudgetLimiter(tpm=2000, clock=clock.time, sleeper=clock.sleep)
    limiter.reserve(1900)

    client = GroqClient(
        api_key="test",
        limiters={"openai/gpt-oss-20b": limiter},
        transport=StubTransport(['{"name": "a", "score": 1.0}']),
        max_wait=5.0,
    )

    with pytest.raises(RateLimitError):
        client.structured(
            model="openai/gpt-oss-20b", system="s", user="u", schema_model=Tiny
        )
    assert clock.slept == []


def test_a_client_without_a_ceiling_still_waits():
    clock = FakeClock()
    limiter = TokenBudgetLimiter(tpm=2000, clock=clock.time, sleeper=clock.sleep)
    limiter.reserve(1900)

    client = GroqClient(
        api_key="test",
        limiters={"openai/gpt-oss-20b": limiter},
        transport=StubTransport(['{"name": "a", "score": 1.0}']),
    )

    client.structured(
        model="openai/gpt-oss-20b", system="s", user="u", schema_model=Tiny
    )
    assert clock.slept, "with no ceiling the client must wait, as the CLI relies on"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --python 3.12 --extra dev pytest tests/test_client.py -k ceiling -v
```

Expected: the first fails with `TypeError: __init__() got an unexpected keyword argument 'max_wait'`.

- [ ] **Step 3: Thread the ceiling through**

In `api/newsninja/analysis/client.py`, change `GroqClient.__init__`:

```python
    def __init__(
        self,
        api_key: str,
        limiters: dict[str, TokenBudgetLimiter] | None = None,
        transport: Transport | None = None,
        max_wait: float | None = None,
    ) -> None:
        self._transport = transport or GroqTransport(api_key)
        self._limiters = limiters or {
            model: TokenBudgetLimiter(tpm) for model, tpm in MODEL_TPM.items()
        }
        # None means "wait as long as it takes", which is right for the CLI and
        # the eval harness. The HTTP service supplies a ceiling so a full budget
        # becomes a 429 rather than a held-open connection.
        self._max_wait = max_wait
        self.usage = Usage()
```

and `_reserve`:

```python
    def _reserve(self, model: str, prompt: str) -> int:
        """Book the prompt *and* the completion the response will cost."""
        estimate = _estimate_tokens(prompt) + EXPECTED_COMPLETION_TOKENS
        return self._limiter_for(model).reserve(estimate, max_wait=self._max_wait)
```

- [ ] **Step 4: Run the tests**

```bash
uv run --python 3.12 --extra dev pytest tests/test_client.py -v
uv run --python 3.12 --extra dev mypy newsninja evals
```

Expected: all pass, mypy `Success`.

- [ ] **Step 5: Commit**

```bash
git add api/newsninja/analysis/client.py api/tests/test_client.py
git commit -m "Let a client carry a wait ceiling to its limiters

The limiter can refuse now, but nothing was asking it to. GroqClient owns
the limiters, so the ceiling travels through it and reaches reserve().

Defaults to None, which is how the CLI and the eval harness keep waiting."
```

---

### Task 3: Lift one topic out of `run_pipeline`

`/analyze` needs fetch-then-extract for a single topic. That loop already exists at `pipeline.py:96-107`. Copying it into the API would create two implementations of the same rule about what counts as a skipped source versus a failed one — the exact distinction the project treats as an invariant.

**Files:**
- Modify: `api/newsninja/pipeline.py`
- Test: `api/tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `TopicResult` dataclass with fields `analysis: ArticleAnalysis`, `source_errors: dict[str, list[str]]`, `skipped_sources: list[str]`.
  - `analyze_topic(topic: str, sources: list[Source], client: StructuredClient, cache: Cache | None = None, limit: int = 8) -> TopicResult`.

- [ ] **Step 1: Write the failing tests**

Append to `api/tests/test_pipeline.py`. `FakeSource`, `FakeClient`, `_article` and `_tts` already exist there — reuse them.

```python
from newsninja.pipeline import analyze_topic


def test_analyze_topic_returns_one_analysis():
    result = analyze_topic(
        "ai", [FakeSource("google_news", [_article()])], FakeClient()
    )
    assert result.analysis.topic == "ai"
    assert result.source_errors == {}
    assert result.skipped_sources == []


def test_analyze_topic_records_a_failing_source_without_aborting():
    result = analyze_topic(
        "ai",
        [
            FakeSource("reddit", error=SourceError("reddit", "401")),
            FakeSource("google_news", [_article()]),
        ],
        FakeClient(),
    )
    assert "reddit" in result.source_errors
    assert result.skipped_sources == []


def test_analyze_topic_keeps_skipped_apart_from_failed():
    """A missing credential and a broken source are different facts.

    Collapsing them would tell a visitor to debug a source that was never
    tried, and hide one they could fix by supplying credentials.
    """
    result = analyze_topic(
        "ai",
        [
            FakeSource("reddit", is_available=False),
            FakeSource("google_news", [_article()]),
        ],
        FakeClient(),
    )
    assert result.skipped_sources == ["reddit"]
    assert result.source_errors == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --python 3.12 --extra dev pytest tests/test_pipeline.py -k analyze_topic -v
```

Expected: 3 errors — `ImportError: cannot import name 'analyze_topic'`.

- [ ] **Step 3: Extract the function and rewire `run_pipeline`**

In `api/newsninja/pipeline.py`, add `ArticleAnalysis` to the models import:

```python
from newsninja.models import Article, ArticleAnalysis, Briefing
```

Add the result type next to `PipelineResult`:

```python
@dataclass
class TopicResult:
    """One topic's analysis plus what went wrong gathering it.

    Separate from ``PipelineResult`` because a single topic has no briefing and
    no audio — folding it in would mean a type whose fields are meaningless
    half the time.
    """

    analysis: ArticleAnalysis
    source_errors: dict[str, list[str]] = field(default_factory=dict)
    skipped_sources: list[str] = field(default_factory=list)
```

Add the function above `run_pipeline`:

```python
def analyze_topic(
    topic: str,
    sources: list[Source],
    client: StructuredClient,
    cache: Cache | None = None,
    limit: int = 8,
) -> TopicResult:
    """Fetch ``topic`` from every available source and extract one analysis.

    Split out of ``run_pipeline`` so the HTTP layer can serve a single topic per
    request without restating the rule that separates a skipped source from a
    failed one.
    """
    articles: list[Article] = []
    source_errors: dict[str, list[str]] = {}
    skipped_sources: list[str] = []

    for source in sources:
        if not source.available():
            if source.name not in skipped_sources:
                skipped_sources.append(source.name)
            continue
        try:
            articles.extend(source.fetch(topic, limit=limit))
        except SourceError as exc:
            source_errors.setdefault(source.name, []).append(str(exc))

    return TopicResult(
        analysis=extract_topic(client, topic, articles, cache=cache),
        source_errors=source_errors,
        skipped_sources=skipped_sources,
    )
```

Then replace the loop inside `run_pipeline` (currently `pipeline.py:96-107`) with:

```python
    for topic in topics:
        outcome = analyze_topic(topic, sources, client, cache=cache, limit=limit)
        analyses.append(outcome.analysis)
        for name, messages in outcome.source_errors.items():
            source_errors.setdefault(name, []).extend(messages)
        for name in outcome.skipped_sources:
            if name not in skipped_sources:
                skipped_sources.append(name)
```

This preserves the original accumulation exactly: every failure message is kept (a source failing on several topics keeps each message), and a skipped source is recorded once across the whole run.

- [ ] **Step 4: Run the pipeline suite**

```bash
uv run --python 3.12 --extra dev pytest tests/test_pipeline.py -v
```

Expected: all pass. **The pre-existing pipeline tests must pass without edits** — they are the regression bar for this refactor. If one fails, the extraction changed behaviour: revert and re-derive rather than adjusting the test.

- [ ] **Step 5: Run the whole suite and type checker**

```bash
uv run --python 3.12 --extra dev pytest -q
uv run --python 3.12 --extra dev mypy newsninja evals
uv run --python 3.12 --extra dev ruff check .
```

Expected: 232 passed, mypy `Success`, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add api/newsninja/pipeline.py api/tests/test_pipeline.py
git commit -m "Lift one topic's fetch and extract into analyze_topic

The HTTP layer serves one topic per request, and the loop it needs already
existed inside run_pipeline. Copying it would have produced two
implementations of the rule separating a skipped source from a failed one,
which is a distinction this project treats as an invariant.

run_pipeline now calls it and accumulates as before: every failure message
kept, each skipped source recorded once for the run. The existing pipeline
tests pass unmodified, which is what makes this a refactor rather than a
rewrite."
```

---

### Task 4: Add the dependencies and the service settings

**Files:**
- Modify: `api/pyproject.toml`
- Modify: `api/newsninja/config.py`
- Test: `api/tests/test_config.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Settings.allowed_origins: list[str]`, `Settings.api_max_wait_seconds: float`, `Settings.rate_limit_per_minute: int`, `Settings.trust_proxy_headers: bool`.

- [ ] **Step 1: Write the failing test**

Append to `api/tests/test_config.py`:

```python
def test_service_settings_have_safe_defaults():
    """Defaults must be safe to deploy without reading the docs first.

    No origins means no cross-origin access rather than any; not trusting proxy
    headers means a spoofed X-Forwarded-For cannot bypass the per-IP window.
    """
    settings = Settings()
    assert settings.allowed_origins == []
    assert settings.trust_proxy_headers is False
    assert settings.api_max_wait_seconds == 5.0
    assert settings.rate_limit_per_minute == 10


def test_allowed_origins_reads_a_json_list_from_the_environment(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", '["https://example.vercel.app"]')
    assert Settings().allowed_origins == ["https://example.vercel.app"]
```

If `Settings` is not already imported at the top of that file, add `from newsninja.config import Settings`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --python 3.12 --extra dev pytest tests/test_config.py -k "service_settings or allowed_origins" -v
```

Expected: `AttributeError: 'Settings' object has no attribute 'allowed_origins'`.

- [ ] **Step 3: Add the dependencies**

In `api/pyproject.toml`, add to `dependencies`:

```toml
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
```

`httpx` is already a dependency and is what Starlette's `TestClient` requires, so no new dev dependency is needed.

- [ ] **Step 4: Add the settings fields**

In `api/newsninja/config.py`, add to `Settings` below `request_timeout`:

```python
    #: Origins permitted to call this service from a browser. Empty by default:
    #: no cross-origin access is a safe failure, "*" is not.
    allowed_origins: list[str] = []
    #: How long a request handler may sit inside the token limiter before the
    #: service answers 429 instead. Keeps a full budget from becoming a
    #: connection held open until the platform proxy severs it.
    api_max_wait_seconds: float = 5.0
    #: Per-IP request ceiling. Stops one client hammering the service; it does
    #: not protect the shared token budget — api_max_wait_seconds does that.
    rate_limit_per_minute: int = 10
    #: Read the client address from X-Forwarded-For. Off by default: the header
    #: is spoofable unless the platform overwrites it, and trusting it blindly
    #: turns the per-IP window into decoration.
    trust_proxy_headers: bool = False
```

- [ ] **Step 5: Sync and run the tests**

```bash
uv sync --python 3.12 --extra dev
uv run --python 3.12 --extra dev pytest tests/test_config.py -v
uv run --python 3.12 --extra dev mypy newsninja evals
```

Expected: all pass, mypy `Success`.

- [ ] **Step 6: Commit**

```bash
git add api/pyproject.toml api/uv.lock api/newsninja/config.py api/tests/test_config.py
git commit -m "Add the service dependencies and its four settings

Defaults are chosen so an unconfigured deployment fails safe: no allowed
origins rather than any, and proxy headers untrusted rather than believed.
A spoofable X-Forwarded-For taken on faith would turn the per-IP window
into decoration."
```

---

### Task 5: Per-IP rate limiting

This stops one client hammering the service. It does **not** protect the token budget — at ~2,400 tokens per analysis against 8,000 TPM the service supports roughly three analyses per minute across all visitors combined, and ten well-behaved callers from ten IPs each pass this check. The ceiling from Task 1 is what catches that. Two defences, two different jobs.

**Files:**
- Create: `api/newsninja/api/__init__.py`
- Create: `api/newsninja/api/ratelimit.py`
- Test: `api/tests/test_api_ratelimit.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `RateLimiter(limit: int, clock: Callable[[], float] = time.monotonic)` with `check(key: str) -> float | None` — `None` when allowed, otherwise seconds until a slot frees.

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p api/newsninja/api
printf '"""HTTP layer over the analysis package."""\n' > api/newsninja/api/__init__.py
git check-ignore -v api/newsninja/api/__init__.py; echo "exit $? — 1 means tracked, good"
```

If `git check-ignore` exits 0, the path is being swallowed by `.gitignore`. Add a scoped negation for `api/newsninja/api/` next to the existing ones rather than broadening any rule, then re-check.

- [ ] **Step 2: Write the failing tests**

Create `api/tests/test_api_ratelimit.py`:

```python
from newsninja.api.ratelimit import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now


def test_requests_under_the_limit_are_allowed():
    clock = FakeClock()
    limiter = RateLimiter(limit=3, clock=clock.time)
    assert [limiter.check("1.2.3.4") for _ in range(3)] == [None, None, None]


def test_the_request_over_the_limit_is_refused_with_a_wait():
    clock = FakeClock()
    limiter = RateLimiter(limit=2, clock=clock.time)
    limiter.check("1.2.3.4")
    clock.now += 10.0
    limiter.check("1.2.3.4")
    assert limiter.check("1.2.3.4") == 50.0


def test_the_window_rolls_forward():
    clock = FakeClock()
    limiter = RateLimiter(limit=1, clock=clock.time)
    limiter.check("1.2.3.4")
    clock.now += 61.0
    assert limiter.check("1.2.3.4") is None


def test_clients_are_counted_separately():
    clock = FakeClock()
    limiter = RateLimiter(limit=1, clock=clock.time)
    limiter.check("1.2.3.4")
    assert limiter.check("5.6.7.8") is None


def test_idle_clients_are_evicted_rather_than_accumulating():
    """Without eviction the key map is an unbounded allocation per address.

    A caller cycling source addresses would grow it without limit, which turns
    the rate limiter into the memory-exhaustion vector it exists to prevent.
    """
    clock = FakeClock()
    limiter = RateLimiter(limit=1, clock=clock.time)
    for octet in range(200):
        limiter.check(f"10.0.0.{octet}")
    clock.now += 61.0
    limiter.check("10.0.1.1")
    assert limiter.tracked_clients() == 1
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
uv run --python 3.12 --extra dev pytest tests/test_api_ratelimit.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'newsninja.api.ratelimit'`.

- [ ] **Step 4: Implement the limiter**

Create `api/newsninja/api/ratelimit.py`:

```python
"""Per-IP request window.

This bounds how fast one client may call the service. It does not protect the
shared token budget — ten callers from ten addresses each pass this check and
still exhaust it. ``TokenBudgetLimiter``'s wait ceiling is what catches that.
"""

import time
from collections import deque
from collections.abc import Callable

WINDOW_SECONDS = 60.0


class RateLimiter:
    """Sliding window of request timestamps, keyed by client address."""

    def __init__(self, limit: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._limit = limit
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def tracked_clients(self) -> int:
        """How many addresses currently hold state. Reporting and tests only."""
        return len(self._hits)

    def _evict_idle(self, now: float) -> None:
        """Drop addresses with nothing left inside the window.

        Without this the key map grows once per distinct address seen and never
        shrinks, so a caller cycling source addresses could exhaust memory
        through the very component meant to bound abuse.
        """
        for key in [k for k, hits in self._hits.items() if not hits]:
            del self._hits[key]

    def check(self, key: str) -> float | None:
        """Record a request for ``key``.

        Returns ``None`` when the request is allowed, otherwise the seconds
        until a slot frees. A refused request is not recorded — counting it
        would extend the block every time a blocked client retried.
        """
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= WINDOW_SECONDS:
            hits.popleft()

        if len(hits) >= self._limit:
            return hits[0] + WINDOW_SECONDS - now

        hits.append(now)
        self._evict_idle(now)
        return None
```

- [ ] **Step 5: Run the tests**

```bash
uv run --python 3.12 --extra dev pytest tests/test_api_ratelimit.py -v
uv run --python 3.12 --extra dev mypy newsninja evals
uv run --python 3.12 --extra dev ruff check .
```

Expected: 5 passed, mypy `Success`, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add api/newsninja/api/__init__.py api/newsninja/api/ratelimit.py api/tests/test_api_ratelimit.py
git commit -m "Add a per-IP request window

Bounds how fast one client may call the service. It deliberately does not
try to protect the token budget: ten callers from ten addresses each pass
this check and still drain it, which is the limiter ceiling's job.

Idle addresses are evicted. Without that the key map grows once per address
seen and never shrinks, so a caller cycling source addresses could exhaust
memory through the component meant to bound abuse."
```

---

### Task 6: The application, its dependencies, and `/health`

**Files:**
- Create: `api/newsninja/api/deps.py`
- Create: `api/newsninja/api/schemas.py`
- Create: `api/newsninja/api/routes.py`
- Create: `api/newsninja/api/app.py`
- Modify: `api/newsninja/api/__init__.py`
- Modify: `api/tests/conftest.py` (adds the shared `api_app` fixture)
- Test: `api/tests/test_api_app.py`

**Interfaces:**
- Consumes: `Settings` fields from Task 4, `RateLimiter` from Task 5, `GroqClient(max_wait=...)` from Task 2.
- Produces:
  - `newsninja.api.deps.get_client() -> GroqClient`, `get_cache() -> Cache`, `get_sources() -> list[Source]`, `get_tts() -> Callable[[str, str, bool], bytes]` — all FastAPI dependencies, all process-singletons.
  - `newsninja.api.schemas`: `AnalyzeRequest`, `AnalyzeResponse`, `BriefRequest`, `BriefResponse`, `AudioRequest`, `HealthResponse`, plus `MAX_SCRIPT_CHARS` and `LANGUAGE_PATTERN`.
  - `newsninja.api.routes.router` — an `APIRouter` carrying every endpoint.
  - `newsninja.api.app.create_app(settings: Settings | None = None) -> FastAPI` and the module-level `app = create_app()`, re-exported as `newsninja.api.app`.

**Why a factory rather than a bare module-level `FastAPI()`:** middleware is configured from settings at construction time. A module-level `add_middleware(CORSMiddleware, allow_origins=get_settings().allowed_origins)` reads the environment once, at import, and no later change to `ALLOWED_ORIGINS` can affect it — so CORS would be untestable and unconfigurable after the first import. A factory also lets the per-IP window live in the app rather than in a module global that leaks between tests.

- [ ] **Step 1: Add a shared application fixture**

Append to `api/tests/conftest.py`:

```python
@pytest.fixture
def api_app():
    """A fresh application per test.

    The per-IP window lives on the application and every TestClient reports the
    same address, so a shared instance would let one module's requests exhaust
    another module's allowance — an ordering bug that looks like a flaky test.
    The limit is raised out of the way here; tests that exercise the window
    build their own app with a real one.
    """
    from newsninja.api.app import create_app
    from newsninja.config import Settings

    return create_app(Settings(rate_limit_per_minute=10_000))
```

- [ ] **Step 2: Write the failing tests**

Create `api/tests/test_api_app.py`:

```python
import pytest
from fastapi.testclient import TestClient

from newsninja.api.deps import get_client


@pytest.fixture
def client(api_app):
    with TestClient(api_app) as test_client:
        yield test_client


def test_health_reports_ok_and_the_package_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_health_never_calls_the_model(api_app, client):
    """A health check that spends tokens against an 8,000 TPM budget causes
    the outages it is supposed to detect."""

    def _explode():
        raise AssertionError("/health must not build or call the model client")

    api_app.dependency_overrides[get_client] = _explode
    assert client.get("/health").status_code == 200


def test_the_model_client_is_one_instance_for_the_process():
    """Silent when broken, so it is asserted directly.

    GroqClient owns the TokenBudgetLimiter. A client built per request would
    hand every request a fresh, empty budget window, the limiter would never
    wait, and the first symptom would be Groq 429s in production.
    """
    get_client.cache_clear()
    assert get_client() is get_client()
    get_client.cache_clear()
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
uv run --python 3.12 --extra dev pytest tests/test_api_app.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'newsninja.api.app'`.

- [ ] **Step 4: Write the request and response schemas**

Create `api/newsninja/api/schemas.py`:

```python
"""HTTP request and response bodies.

Kept apart from ``newsninja.models``: those are the domain contract and the
LLM output schema, and letting HTTP concerns leak into them would couple the
wire format to the thing being measured.
"""

from pydantic import BaseModel, Field, field_validator

from newsninja.models import ArticleAnalysis, Briefing
from newsninja.pipeline import MAX_TOPICS

#: A language code reaches a translation prompt, so it is not free text.
LANGUAGE_PATTERN = r"^[a-z]{2}(-[A-Za-z]{2})?$"

#: chunk_text splits at 3,000 characters and gTTS makes one outbound call per
#: chunk, so an uncapped script is an amplification vector: 1 MB of text becomes
#: roughly 350 requests leaving the host.
MAX_SCRIPT_CHARS = 20_000


class AnalyzeRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=8, ge=1, le=12)

    @field_validator("topic")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("topic must not be blank")
        return stripped


class AnalyzeResponse(BaseModel):
    analysis: ArticleAnalysis
    #: Sources that were tried and broke, by name.
    source_errors: dict[str, list[str]] = {}
    #: Sources that reported themselves unavailable and were never tried.
    #: Separate from source_errors on purpose: a missing credential is
    #: something the caller can fix, a broken source is not.
    skipped_sources: list[str] = []


class BriefRequest(BaseModel):
    analyses: list[ArticleAnalysis] = Field(min_length=1, max_length=MAX_TOPICS)
    language: str = Field(default="en", pattern=LANGUAGE_PATTERN)


class BriefResponse(BaseModel):
    briefing: Briefing


class AudioRequest(BaseModel):
    script: str = Field(min_length=1, max_length=MAX_SCRIPT_CHARS)
    language: str = Field(default="en", pattern=LANGUAGE_PATTERN)


class HealthResponse(BaseModel):
    status: str
    version: str
```

- [ ] **Step 5: Write the dependencies**

Create `api/newsninja/api/deps.py`:

```python
"""Process-wide singletons, supplied to handlers as FastAPI dependencies.

``lru_cache`` is what makes them singletons, and that is load-bearing rather
than an optimisation. ``GroqClient`` owns the ``TokenBudgetLimiter``; building
one per request would give every request a fresh, empty budget window, so the
limiter would never wait and the service would walk straight into Groq 429s
with no local warning at all.
"""

from collections.abc import Callable
from functools import lru_cache

from newsninja.analysis.client import GroqClient
from newsninja.cache import Cache
from newsninja.config import get_settings
from newsninja.pipeline import default_tts
from newsninja.sources.base import Source
from newsninja.sources.google_news import GoogleNewsSource
from newsninja.sources.reddit import RedditSource


@lru_cache(maxsize=1)
def get_client() -> GroqClient:
    settings = get_settings()
    return GroqClient(
        api_key=settings.groq_api_key,
        max_wait=settings.api_max_wait_seconds,
    )


@lru_cache(maxsize=1)
def get_cache() -> Cache:
    return Cache(get_settings().cache_path)


@lru_cache(maxsize=1)
def get_sources() -> list[Source]:
    settings = get_settings()
    return [
        GoogleNewsSource(timeout=settings.request_timeout),
        RedditSource(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        ),
    ]


@lru_cache(maxsize=1)
def get_tts() -> Callable[[str, str, bool], bytes]:
    # Orpheus is a separate REST endpoint rather than part of the chat client,
    # so the key has to reach the speech seam too.
    return default_tts(get_settings().groq_api_key)
```

- [ ] **Step 6: Write the routes module with `/health`**

Create `api/newsninja/api/routes.py`:

```python
"""The endpoints.

Split along seams the package already has, so no single request runs long
enough for a platform proxy to sever it. See
docs/design/specs/2026-07-31-fastapi-service-design.md for the token
arithmetic that forces this shape.

Every handler is a plain ``def`` rather than ``async def``, so FastAPI runs it
in a threadpool. The token limiter blocks with ``time.sleep``; on the event
loop that would stall every other request in the process.
"""

from importlib.metadata import version

from fastapi import APIRouter

from newsninja.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only.

    Deliberately does not call Groq. A health check that spends tokens against
    an 8,000 TPM budget is a health check that causes outages.
    """
    return HealthResponse(status="ok", version=version("newsninja"))
```

- [ ] **Step 7: Write the application factory**

Create `api/newsninja/api/app.py`:

```python
"""Application assembly.

``create_app`` exists because middleware is configured from settings at
construction time. Adding CORS at module scope would read the environment once,
at import, and no later configuration could change it — which makes the
allowlist both untestable and unchangeable after the first import.
"""

from importlib.metadata import version

from fastapi import FastAPI

from newsninja.api.routes import router
from newsninja.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the service. Pass ``settings`` to override the environment."""
    resolved = settings if settings is not None else get_settings()

    application = FastAPI(
        title="NewsNinja",
        description="Source-grounded news briefings with structured extraction.",
        version=version("newsninja"),
    )
    application.state.settings = resolved
    application.include_router(router)
    return application


app = create_app()
```

Replace `api/newsninja/api/__init__.py` with:

```python
"""HTTP layer over the analysis package."""

from newsninja.api.app import app, create_app

__all__ = ["app", "create_app"]
```

- [ ] **Step 8: Run the tests**

```bash
uv run --python 3.12 --extra dev pytest tests/test_api_app.py -v
uv run --python 3.12 --extra dev mypy newsninja evals
uv run --python 3.12 --extra dev ruff check .
```

Expected: 3 passed, mypy `Success`, ruff clean.

- [ ] **Step 9: Commit**

```bash
git add api/newsninja/api/ api/tests/conftest.py api/tests/test_api_app.py
git commit -m "Add the FastAPI application, its singletons, and /health

The singletons are load-bearing rather than an optimisation. GroqClient owns
the token limiter, so a client built per request would hand every request an
empty budget window; the limiter would never wait and the first symptom
would be Groq 429s in production. That failure is silent, so it gets a test.

/health reports liveness and the package version read from metadata rather
than a literal, and never calls the model. A health check that spends tokens
against an 8,000 TPM budget causes the outages it exists to detect."
```

---

### Task 7: `POST /analyze`

**Files:**
- Modify: `api/newsninja/api/routes.py`
- Test: `api/tests/test_api_analyze.py`

**Interfaces:**
- Consumes: `analyze_topic`/`TopicResult` (Task 3), `AnalyzeRequest`/`AnalyzeResponse` (Task 6), `get_client`/`get_cache`/`get_sources` (Task 6).
- Produces: `POST /analyze`.

- [ ] **Step 1: Write the failing tests**

Create `api/tests/test_api_analyze.py`:

```python
import pytest
from fastapi.testclient import TestClient

from newsninja.api.deps import get_cache, get_client, get_sources
from newsninja.errors import SourceError
from newsninja.models import Article, ArticleAnalysis


class FakeSource:
    def __init__(self, name, articles=None, error=None, is_available=True):
        self.name = name
        self._articles = articles or []
        self._error = error
        self._available = is_available

    def available(self):
        return self._available

    def fetch(self, topic, limit=8):
        if self._error:
            raise self._error
        return self._articles


class FakeClient:
    def structured(self, *, model, system, user, schema_model, max_retries=2):
        return ArticleAnalysis(
            topic="ai", summary="s", entities=[], stance="neutral",
            confidence=0.5, key_claims=[],
        )

    def text(self, *, model, system, user):
        return "Here is your briefing."


def _article():
    return Article(title="T", url="u", source="google_news", body="b")


@pytest.fixture
def client(api_app):
    api_app.dependency_overrides[get_client] = FakeClient
    api_app.dependency_overrides[get_cache] = lambda: None
    api_app.dependency_overrides[get_sources] = lambda: [
        FakeSource("google_news", [_article()])
    ]
    with TestClient(api_app) as test_client:
        yield test_client


def test_analyze_returns_one_analysis(client):
    response = client.post("/analyze", json={"topic": "ai"})
    assert response.status_code == 200
    body = response.json()
    assert body["analysis"]["topic"] == "ai"
    assert body["source_errors"] == {}
    assert body["skipped_sources"] == []


def test_a_degraded_run_is_still_a_success(api_app, client):
    """A briefing built from fewer sources is a result, not a failure.

    run_pipeline already treats it that way; the HTTP layer must not disagree
    with the package it wraps.
    """
    api_app.dependency_overrides[get_sources] = lambda: [
        FakeSource("reddit", error=SourceError("reddit", "401")),
        FakeSource("google_news", [_article()]),
    ]
    response = client.post("/analyze", json={"topic": "ai"})
    assert response.status_code == 200
    assert "reddit" in response.json()["source_errors"]


def test_skipped_and_failed_sources_stay_in_separate_fields(api_app, client):
    api_app.dependency_overrides[get_sources] = lambda: [
        FakeSource("reddit", is_available=False),
        FakeSource("google_news", [_article()]),
    ]
    body = client.post("/analyze", json={"topic": "ai"}).json()
    assert body["skipped_sources"] == ["reddit"]
    assert body["source_errors"] == {}


@pytest.mark.parametrize(
    "payload",
    [
        {"topic": "   "},
        {"topic": ""},
        {"topic": "x" * 201},
        {"topic": "ai", "limit": 0},
        {"topic": "ai", "limit": 13},
    ],
)
def test_invalid_requests_are_refused(client, payload):
    assert client.post("/analyze", json=payload).status_code == 422
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --python 3.12 --extra dev pytest tests/test_api_analyze.py -v
```

Expected: all fail with 404 — the route does not exist yet.

- [ ] **Step 3: Add the endpoint**

In `api/newsninja/api/routes.py`, extend the imports:

```python
from typing import Annotated

from fastapi import APIRouter, Depends

from newsninja.analysis.client import GroqClient
from newsninja.api.deps import get_cache, get_client, get_sources
from newsninja.api.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
from newsninja.cache import Cache
from newsninja.pipeline import analyze_topic
from newsninja.sources.base import Source
```

and add below `health`:

```python
@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    payload: AnalyzeRequest,
    client: Annotated[GroqClient, Depends(get_client)],
    cache: Annotated[Cache | None, Depends(get_cache)],
    sources: Annotated[list[Source], Depends(get_sources)],
) -> AnalyzeResponse:
    """Fetch and extract a single topic.

    One topic per request is what keeps this short. Five topics reserve 11,787
    tokens against an 8,000 TPM ceiling, so the fourth would sit in the limiter
    for 48 seconds and the connection would not survive it.
    """
    outcome = analyze_topic(
        payload.topic, sources, client, cache=cache, limit=payload.limit
    )
    return AnalyzeResponse(
        analysis=outcome.analysis,
        source_errors=outcome.source_errors,
        skipped_sources=outcome.skipped_sources,
    )
```

This is a `def`, not an `async def`, so FastAPI runs it in a threadpool and the limiter's blocking sleep does not stall the event loop.

- [ ] **Step 4: Run the tests**

```bash
uv run --python 3.12 --extra dev pytest tests/test_api_analyze.py -v
uv run --python 3.12 --extra dev mypy newsninja evals
```

Expected: 8 passed, mypy `Success`.

- [ ] **Step 5: Commit**

```bash
git add api/newsninja/api/routes.py api/tests/test_api_analyze.py
git commit -m "Add POST /analyze for a single topic

One topic per request is what keeps the request short. Five topics reserve
11,787 tokens against an 8,000 TPM ceiling, so the fourth sits in the
limiter for 48 seconds and no platform proxy will hold the connection.

A degraded run still returns 200, and skipped sources stay in a different
field from failed ones. Both match how run_pipeline already behaves; the
HTTP layer must not disagree with the package it wraps."
```

---

### Task 8: `POST /brief`

**Files:**
- Modify: `api/newsninja/api/routes.py`
- Test: `api/tests/test_api_brief.py`

**Interfaces:**
- Consumes: `BriefRequest`/`BriefResponse` (Task 6), `get_client` (Task 6), `build_briefing` (existing, `newsninja.analysis.synthesize`).
- Produces: `POST /brief`.

- [ ] **Step 1: Write the failing tests**

Create `api/tests/test_api_brief.py`:

```python
import pytest
from fastapi.testclient import TestClient

from newsninja.api.deps import get_client


class FakeClient:
    def structured(self, *, model, system, user, schema_model, max_retries=2):
        raise AssertionError("/brief must not run extraction")

    def text(self, *, model, system, user):
        return "Here is your briefing."


def _analysis(topic="ai"):
    return {
        "topic": topic,
        "summary": "s",
        "entities": [],
        "stance": "neutral",
        "confidence": 0.5,
        "key_claims": [],
    }


@pytest.fixture
def client(api_app):
    api_app.dependency_overrides[get_client] = FakeClient
    with TestClient(api_app) as test_client:
        yield test_client


def test_brief_synthesises_across_every_analysis(client):
    """The unified script is the product. Synthesising per batch would yield
    two disconnected briefings for five topics, which is the reason this
    endpoint takes all the analyses at once."""
    payload = {"analyses": [_analysis("ai"), _analysis("climate")], "language": "en"}
    response = client.post("/brief", json=payload)
    assert response.status_code == 200
    briefing = response.json()["briefing"]
    assert briefing["script"] == "Here is your briefing."
    assert briefing["topics"] == ["ai", "climate"]


def test_more_than_five_analyses_are_refused(client):
    payload = {"analyses": [_analysis(f"t{n}") for n in range(6)]}
    assert client.post("/brief", json=payload).status_code == 422


def test_no_analyses_is_refused(client):
    assert client.post("/brief", json={"analyses": []}).status_code == 422


@pytest.mark.parametrize("language", ["english", "e", "EN", "en_GB", "'; DROP"])
def test_a_language_code_is_not_free_text(client, language):
    """It reaches a translation prompt, so it is validated rather than trusted."""
    payload = {"analyses": [_analysis()], "language": language}
    assert client.post("/brief", json=payload).status_code == 422
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --python 3.12 --extra dev pytest tests/test_api_brief.py -v
```

Expected: all fail with 404.

- [ ] **Step 3: Add the endpoint**

In `api/newsninja/api/routes.py`, extend the imports:

```python
from newsninja.analysis.synthesize import build_briefing
from newsninja.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BriefRequest,
    BriefResponse,
    HealthResponse,
)
```

and add:

```python
@router.post("/brief", response_model=BriefResponse)
def brief(
    payload: BriefRequest,
    client: Annotated[GroqClient, Depends(get_client)],
) -> BriefResponse:
    """Synthesise one script across every supplied analysis.

    This renders whatever it is handed. The server holds no articles at this
    point and cannot re-check that quotes are verbatim, so the grounding
    guarantee belongs to /analyze, which produced them.
    """
    briefing = build_briefing(client, payload.analyses, language=payload.language)
    return BriefResponse(briefing=briefing)
```

- [ ] **Step 4: Run the tests**

```bash
uv run --python 3.12 --extra dev pytest tests/test_api_brief.py -v
uv run --python 3.12 --extra dev mypy newsninja evals
```

Expected: 8 passed, mypy `Success`.

- [ ] **Step 5: Commit**

```bash
git add api/newsninja/api/routes.py api/tests/test_api_brief.py
git commit -m "Add POST /brief over all the supplied analyses

Taking every analysis at once is what preserves the unified script.
Synthesising per batch would hand back two disconnected briefings for five
topics, and the single script is the product.

The endpoint renders what it is given. It holds no articles and cannot
re-check that quotes are verbatim, so the grounding guarantee stays with
/analyze, which produced them."
```

---

### Task 9: `POST /audio`

**Files:**
- Modify: `api/newsninja/api/routes.py`
- Test: `api/tests/test_api_audio.py`

**Interfaces:**
- Consumes: `AudioRequest`/`MAX_SCRIPT_CHARS` (Task 6), `get_tts` (Task 6).
- Produces: `POST /audio`, returning raw bytes rather than JSON.

- [ ] **Step 1: Write the failing tests**

Create `api/tests/test_api_audio.py`:

```python
import pytest
from fastapi.testclient import TestClient

from newsninja.api.deps import get_tts
from newsninja.api.schemas import MAX_SCRIPT_CHARS
from newsninja.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """get_settings is lru_cached, so a monkeypatched variable is invisible
    until the cache is dropped — before the test as well as after it."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(api_app):
    api_app.dependency_overrides[get_tts] = lambda: (
        lambda text, language, enable_orpheus: b"AUDIO"
    )
    with TestClient(api_app) as test_client:
        yield test_client


def test_audio_returns_bytes_not_json(client):
    response = client.post("/audio", json={"script": "hello", "language": "en"})
    assert response.status_code == 200
    assert response.content == b"AUDIO"
    assert response.headers["content-type"] == "audio/mpeg"


def test_the_content_type_follows_the_configured_engine(client, monkeypatch):
    """gTTS returns mp3 and Orpheus returns wav, so the header cannot be a
    constant without lying about one of them."""
    monkeypatch.setenv("ENABLE_ORPHEUS", "true")
    get_settings.cache_clear()
    response = client.post("/audio", json={"script": "hello"})
    assert response.headers["content-type"] == "audio/wav"


def test_an_oversized_script_is_refused(client):
    """chunk_text splits at 3,000 characters and gTTS makes one outbound call
    per chunk, so an uncapped script is an amplification vector."""
    payload = {"script": "x" * (MAX_SCRIPT_CHARS + 1)}
    assert client.post("/audio", json=payload).status_code == 422


def test_an_empty_script_is_refused(client):
    assert client.post("/audio", json={"script": ""}).status_code == 422
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --python 3.12 --extra dev pytest tests/test_api_audio.py -v
```

Expected: all fail with 404.

- [ ] **Step 3: Add the endpoint**

In `api/newsninja/api/routes.py`, extend the imports:

```python
from collections.abc import Callable

from fastapi import APIRouter, Depends, Response

from newsninja.api.deps import get_cache, get_client, get_sources, get_tts
from newsninja.config import Settings, get_settings
```

Add `AudioRequest` to the `newsninja.api.schemas` import list, then add:

```python
@router.post("/audio")
def audio(
    payload: AudioRequest,
    tts: Annotated[Callable[[str, str, bool], bytes], Depends(get_tts)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Render a supplied script to speech.

    Returns raw bytes rather than JSON. The media type follows the configured
    engine because synthesize_speech returns mp3 from gTTS and wav from
    Orpheus; a constant header would misdescribe one of them.
    """
    spoken = tts(payload.script, payload.language, settings.enable_orpheus)
    media_type = "audio/wav" if settings.enable_orpheus else "audio/mpeg"
    return Response(content=spoken, media_type=media_type)
```

- [ ] **Step 4: Run the tests**

```bash
uv run --python 3.12 --extra dev pytest tests/test_api_audio.py -v
uv run --python 3.12 --extra dev mypy newsninja evals
```

Expected: 4 passed, mypy `Success`.

- [ ] **Step 5: Commit**

```bash
git add api/newsninja/api/routes.py api/tests/test_api_audio.py
git commit -m "Add POST /audio for a supplied script

The script is capped at 20,000 characters. chunk_text splits at 3,000 and
gTTS makes one outbound call per chunk, so an uncapped script turns the
service into an amplifier: a megabyte of text becomes roughly 350 requests
leaving the host.

The media type follows the configured engine rather than being a constant,
because gTTS returns mp3 and Orpheus returns wav."
```

---

### Task 10: Error mapping, CORS, the rate-limit middleware, and the README

The last task wires the three cross-cutting concerns and documents the service. They ship together because none of them is independently reviewable: an error envelope with no CORS exposure is unreadable from a browser, and a rate limiter with no documented limits is a trap.

**Files:**
- Modify: `api/newsninja/api/app.py`
- Modify: `api/README.md`
- Test: `api/tests/test_api_errors.py`

**Interfaces:**
- Consumes: everything above.
- Produces: exception handlers for `RateLimitError`, `SourceError`, `ExtractionFailure`, `ValueError`; CORS middleware; per-IP middleware.

- [ ] **Step 1: Write the failing tests**

Create `api/tests/test_api_errors.py`:

```python
import pytest
from fastapi.testclient import TestClient

from newsninja.api.app import create_app
from newsninja.api.deps import get_cache, get_client, get_sources
from newsninja.config import Settings, get_settings
from newsninja.errors import ExtractionFailure, RateLimitError, SourceError
from newsninja.models import Article


def _sources():
    return [_OneArticleSource()]


class _OneArticleSource:
    name = "google_news"

    def available(self):
        return True

    def fetch(self, topic, limit=8):
        return [Article(title="T", url="u", source="google_news", body="b")]


def _raising_client(exc):
    class Raising:
        def structured(self, *, model, system, user, schema_model, max_retries=2):
            raise exc

        def text(self, *, model, system, user):
            raise exc

    return Raising


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """get_settings is lru_cached, so a monkeypatched variable stays invisible
    until the cache is dropped — before the test as well as after it."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(api_app):
    api_app.dependency_overrides[get_cache] = lambda: None
    api_app.dependency_overrides[get_sources] = _sources
    with TestClient(api_app) as test_client:
        yield test_client


def test_a_rate_limit_becomes_429_with_a_retry_after_header(api_app, client):
    """The header is what proxies and browsers honour; the body is what the
    frontend can render without reading headers through CORS. Both, not one."""
    api_app.dependency_overrides[get_client] = _raising_client(
        RateLimitError("budget full", retry_after=48.0)
    )
    response = client.post("/analyze", json={"topic": "ai"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "48"
    body = response.json()["error"]
    assert body["type"] == "rate_limit"
    assert body["retry_after"] == 48.0


def test_retry_after_rounds_up_rather_than_truncating(api_app, client):
    """Truncating 0.4s to "0" tells the caller to retry immediately into a
    budget that is still full."""
    api_app.dependency_overrides[get_client] = _raising_client(
        RateLimitError("budget full", retry_after=0.4)
    )
    response = client.post("/analyze", json={"topic": "ai"})
    assert response.headers["retry-after"] == "1"


def test_an_extraction_failure_becomes_502(api_app, client):
    api_app.dependency_overrides[get_client] = _raising_client(
        ExtractionFailure("did not validate")
    )
    response = client.post("/analyze", json={"topic": "ai"})
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "extraction_failure"


def test_a_source_error_becomes_502_and_names_the_source(api_app, client):
    api_app.dependency_overrides[get_client] = _raising_client(
        SourceError("reddit", "401")
    )
    response = client.post("/analyze", json={"topic": "ai"})
    assert response.status_code == 502
    assert "reddit" in response.json()["error"]["message"]


def test_the_rate_limit_window_refuses_a_hammering_client():
    """The window counts requests, not successes.

    An empty body is used deliberately: it is refused at validation, so the
    handler never runs and no source is ever fetched. A payload that reached
    the real sources would put this test on the network.
    """
    limited = create_app(Settings(rate_limit_per_minute=2))
    with TestClient(limited) as test_client:
        assert test_client.post("/analyze", json={}).status_code == 422
        assert test_client.post("/analyze", json={}).status_code == 422
        response = test_client.post("/analyze", json={})
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit"


def test_health_is_exempt_from_the_window():
    """Uptime pings must not consume a visitor's allowance."""
    limited = create_app(Settings(rate_limit_per_minute=1))
    with TestClient(limited) as test_client:
        codes = [test_client.get("/health").status_code for _ in range(5)]
    assert codes == [200] * 5


def test_retry_after_is_readable_by_a_browser():
    """CORS hides all but a six-header safelist from JavaScript, and
    Retry-After is not on it. Without expose_headers the frontend receives the
    429 and cannot read how long to wait."""
    origin = "https://example.vercel.app"
    cors = create_app(Settings(allowed_origins=[origin]))
    with TestClient(cors) as test_client:
        response = test_client.get("/health", headers={"Origin": origin})
    assert "Retry-After" in response.headers["access-control-expose-headers"]


def test_an_unlisted_origin_is_not_granted_access():
    """An empty allowlist must mean no origin, not any origin."""
    cors = create_app(Settings(allowed_origins=["https://example.vercel.app"]))
    with TestClient(cors) as test_client:
        response = test_client.get("/health", headers={"Origin": "https://evil.test"})
    assert "access-control-allow-origin" not in response.headers
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --python 3.12 --extra dev pytest tests/test_api_errors.py -v
```

Expected: failures — the handlers and the middleware do not exist, so errors surface as 500s and the CORS assertions raise `KeyError`.

- [ ] **Step 3: Add the handlers, CORS, and the window inside the factory**

Everything here goes **inside `create_app`**, not at module scope. Middleware and the per-IP window are both built from settings; at module scope they would be frozen at import time, which is the reason the factory exists.

In `api/newsninja/api/app.py`, extend the imports:

```python
import math
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from newsninja.api.ratelimit import RateLimiter
from newsninja.api.routes import router
from newsninja.config import Settings, get_settings
from newsninja.errors import ExtractionFailure, RateLimitError, SourceError
```

Add this module-level helper — it depends on no settings, so it does not belong in the factory:

```python
def _error(status: int, kind: str, message: str, retry_after: float | None = None) -> JSONResponse:
    """One envelope for every failure, so a caller parses one shape."""
    body: dict[str, object] = {"type": kind, "message": message}
    # Both annotations are required: mypy --strict rejects a bare `{}`.
    headers: dict[str, str] = {}
    if retry_after is not None:
        body["retry_after"] = retry_after
        # delta-seconds is an integer, and rounding up matters: truncating 0.4
        # to "0" tells the caller to retry straight back into a full budget.
        headers["Retry-After"] = str(math.ceil(retry_after))
    return JSONResponse(status_code=status, content={"error": body}, headers=headers)


async def _handle_rate_limit(request: Request, exc: Exception) -> JSONResponse:
    # Typed as Exception because that is the signature Starlette's registry
    # declares. Narrowing in the parameter list would need a `type: ignore`,
    # and this project keeps that count at zero.
    retry_after = exc.retry_after if isinstance(exc, RateLimitError) else None
    return _error(429, "rate_limit", str(exc), retry_after=retry_after)


async def _handle_source_error(request: Request, exc: Exception) -> JSONResponse:
    return _error(502, "source_error", str(exc))


async def _handle_extraction_failure(request: Request, exc: Exception) -> JSONResponse:
    return _error(502, "extraction_failure", str(exc))


async def _handle_value_error(request: Request, exc: Exception) -> JSONResponse:
    return _error(422, "invalid_request", str(exc))


def _client_key(request: Request, trust_proxy: bool) -> str:
    """Identify the caller.

    Behind a proxy, request.client.host is the proxy and every visitor shares
    one bucket. X-Forwarded-For fixes that but is spoofable unless the platform
    overwrites it, so reading it is opt-in rather than automatic.
    """
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client is not None else "unknown"
```

Then extend `create_app` so the whole assembly happens in one place. Replace the body written in Task 6 with:

```python
def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the service. Pass ``settings`` to override the environment."""
    resolved = settings if settings is not None else get_settings()

    application = FastAPI(
        title="NewsNinja",
        description="Source-grounded news briefings with structured extraction.",
        version=version("newsninja"),
    )
    application.state.settings = resolved
    application.include_router(router)

    application.add_exception_handler(RateLimitError, _handle_rate_limit)
    application.add_exception_handler(SourceError, _handle_source_error)
    application.add_exception_handler(ExtractionFailure, _handle_extraction_failure)
    application.add_exception_handler(ValueError, _handle_value_error)

    # One window per application, so a test building a fresh app gets a fresh
    # window rather than inheriting counts from whatever ran before it.
    limiter = RateLimiter(limit=resolved.rate_limit_per_minute)

    @application.middleware("http")
    async def _rate_limit(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # /health is exempt so uptime pings do not consume a visitor's allowance.
        if request.url.path == "/health":
            return await call_next(request)

        wait = limiter.check(_client_key(request, resolved.trust_proxy_headers))
        if wait is not None:
            return _error(429, "rate_limit", "too many requests", retry_after=wait)
        return await call_next(request)

    # Added last, so it wraps the rate-limit middleware and a 429 refused there
    # still carries CORS headers. A 429 a browser cannot read is a 429 the
    # frontend reports as a network error.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        # CORS hides all but a six-header safelist from browser JavaScript, and
        # Retry-After is not on it. Without this the frontend gets the 429 and
        # cannot read how long to wait.
        expose_headers=["Retry-After"],
    )

    return application
```

- [ ] **Step 4: Run the tests**

```bash
uv run --python 3.12 --extra dev pytest tests/test_api_errors.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Document the service in the API README**

Add a section to `api/README.md`. Every claim below must be true of the code as written — if one is not, fix the code rather than softening the sentence.

````markdown
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
`docs/design/specs/2026-07-31-fastapi-service-design.md`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness and package version. Never calls the model. |
| `POST /analyze` | One topic in, one `ArticleAnalysis` out. |
| `POST /brief` | 1–5 analyses in, one unified script out. |
| `POST /audio` | A script in, audio bytes out. |

A client makes N `/analyze` calls, then one `/brief`, then one `/audio`.

### Failures

| Condition | Status |
| --- | --- |
| invalid request body | 422 |
| token budget exhausted | 429, with `Retry-After` |
| a source failed | 502 |
| the model would not produce valid output | 502 |

Every failure uses one envelope:

```json
{ "error": { "type": "rate_limit", "message": "…", "retry_after": 48.0 } }
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

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | required | — |
| `ALLOWED_ORIGINS` | `[]` | JSON list of browser origins |
| `API_MAX_WAIT_SECONDS` | `5.0` | limiter wait before answering 429 |
| `RATE_LIMIT_PER_MINUTE` | `10` | per-IP request ceiling |
| `TRUST_PROXY_HEADERS` | `false` | read `X-Forwarded-For` |
````

- [ ] **Step 6: Run everything**

```bash
uv run --python 3.12 --extra dev pytest -q
uv run --python 3.12 --extra dev mypy newsninja evals
uv run --python 3.12 --extra dev ruff check .
grep -rn "# noqa" newsninja evals | wc -l   # must print 1
grep -rn "type: ignore" newsninja evals     # must print nothing
```

Expected: all tests pass, mypy `Success`, ruff clean, exactly one `# noqa`, no `type: ignore`.

- [ ] **Step 7: Verify the service actually starts**

A passing test suite does not prove the app boots under a real server — `TestClient` and `uvicorn` load it differently.

```bash
uv run --python 3.12 uvicorn newsninja.api:create_app --factory --port 8123 &
sleep 3
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8123/health   # expect 200
curl -s http://127.0.0.1:8123/openapi.json | head -c 200                 # expect JSON
kill %1
```

- [ ] **Step 8: Commit**

```bash
git add api/newsninja/api/app.py api/tests/test_api_errors.py api/README.md
git commit -m "Map the typed errors to statuses, bound callers, and document it

One envelope for every failure so a caller parses one shape. Retry-After
goes in both the header and the body: the header is what proxies honour,
the body is what a browser can read. It rounds up rather than truncating,
because a 0.4s wait reported as 0 sends the caller straight back into a
budget that is still full.

Retry-After is also named in expose_headers. CORS hides all but six headers
from JavaScript and this is not one of them, so without that line the
frontend receives the 429 and cannot tell how long to wait.

/health is exempt from the per-IP window so uptime pings do not consume a
visitor's allowance. The README states what the service does not do: /brief
cannot verify quotes it has no articles for, the window is in-process and
resets on restart, and X-Forwarded-For is spoofable unless the platform
overwrites it."
```

---

## After the plan

Once every task is committed:

1. **Mutation-test the new tests.** The process note in the handoff records six mutations surviving a suite that looked green, so "a test exists" is not evidence. At minimum, mutate: the `max_wait` comparison in `_wait_or_refuse` (`>` to `>=`, then to `<`), `math.ceil` to `int` in `_error`, the `>=` in `RateLimiter.check`, and the `/health` exemption in the middleware. Each mutation must fail at least one test. Write backups to `/tmp`, **never into the repository** — a stray file with a quote in its name was committed once and would have broken `git clone` on Windows.

2. **Update `docs/HANDOFF.md`:** mark Plan 3 complete in §3, and replace the §7 design notes with a pointer to the spec and this plan.

3. **`.github/workflows/ci.yml:22` runs `mypy newsninja` only.** The new `newsninja/api/` package is covered by that, but `evals/` — added by Plan 2 — is not. Worth fixing in the same pass: change it to `mypy newsninja evals` so CI enforces the typing guarantee the project claims.

4. **Still blocked on the owner** and unchanged by this plan: rotate the Groq key, correct the golden labels, and open the Plan 2 PR.
