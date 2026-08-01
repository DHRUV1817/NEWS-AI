# NewsNinja Core Analysis Package — Implementation Plan

> Execute this plan task by task, reviewing each task before starting the next.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Python package that turns a list of topics into a structured, source-grounded news briefing with audio, running entirely on free-tier services.

**Architecture:** A `newsninja` package with four seams — `sources/` (fetch articles), `analysis/` (LLM extraction and synthesis behind one Groq client), `audio/` (TTS), and `cache.py` (SQLite). Nothing outside `analysis/client.py` touches the Groq SDK, so provider swaps and test mocking change one file. A `pipeline.py` orchestrates, and `cli.py` makes the whole thing runnable without a web server.

**Tech Stack:** Python 3.12, Pydantic v2, pydantic-settings, groq, praw, feedparser, gTTS, pytest, respx, uv.

This is **Plan 1 of 4**. Later plans: eval harness, FastAPI service, Next.js frontend. Source spec: `docs/design/specs/2026-07-31-newsninja-portfolio-design.md`.

## Global Constraints

- **Python 3.12.** System Python is 3.9.6 and cannot run `groq`, `praw`, `pydantic-settings`, or `pytest` (all require ≥3.10). Use `uv`, which already has cpython-3.12.13 installed.
- **All work happens in `api/`.** The repository root is reserved for `web/` (Plan 4) and docs.
- **Never call the Groq SDK outside `newsninja/analysis/client.py`.**
- **No network in tests.** Every test uses a recorded fixture or a stub. CI must pass offline.
- **Strict JSON schema requires a transform.** `additionalProperties: false` on every object including `$defs`, and every property listed in `required`. Pydantic emits neither. Verified against the live API 2026-07-31.
- **Structured calls use only `openai/gpt-oss-20b` or `openai/gpt-oss-120b`.** Llama models reject `json_schema` — verified.
- **Free-tier TPM ceilings:** gpt-oss-20b 8,000 · gpt-oss-120b 8,000 · llama-3.3-70b-versatile 12,000.
- **No bare `except:`.** No exception handler may return a user-facing string in place of raising.
- **Every `Claim` carries a `quote` that appears verbatim in its source article.** This is what makes hallucination mechanically detectable.

---

### Task 1: Remove legacy implementation and scaffold the package

The spec (§2) approves deleting every current source file. Doing it first prevents the new package from accidentally importing dead code.

**Files:**
- Delete: `backend.py`, `frontend.py`, `single_file.py`, `start.py`, `streamlit_app.py`, `utils.py`, `news_scraper.py`, `reddit_scraper.py`, `config.py`, `models.py`, `requirements.txt`, `ai-journalist.pdf`, `services/`
- Create: `api/pyproject.toml`, `api/newsninja/__init__.py`, `api/newsninja/config.py`, `api/tests/test_config.py`, `api/tests/conftest.py`

**Interfaces:**
- Consumes: nothing
- Produces: `newsninja.config.Settings` with fields `groq_api_key: str`, `reddit_client_id: str | None`, `reddit_client_secret: str | None`, `reddit_user_agent: str`, `enable_orpheus: bool`, `cache_path: Path`, `request_timeout: int`; and `get_settings() -> Settings`

- [ ] **Step 1: Delete the legacy implementation**

```bash
cd /Users/dhruv/Personal/NEWS-AI
git rm -r --quiet backend.py frontend.py single_file.py start.py streamlit_app.py \
  utils.py news_scraper.py reddit_scraper.py config.py models.py requirements.txt \
  ai-journalist.pdf services/
git status --short
```

Expected: twelve `D` entries plus the four files under `services/`.

- [ ] **Step 2: Create the package skeleton**

```bash
mkdir -p api/newsninja api/tests
touch api/newsninja/__init__.py
```

Write `api/pyproject.toml`:

```toml
[project]
name = "newsninja"
version = "3.0.0"
description = "Source-grounded AI news briefings with structured extraction and evaluation"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.13",
    "pydantic-settings>=2.14",
    "groq>=1.6",
    "feedparser>=6.0.14",
    "praw>=8.0",
    "gtts>=2.5.4",
    "httpx>=0.28",
]

[project.optional-dependencies]
dev = ["pytest>=9.0", "pytest-cov>=5.0", "respx>=0.23", "ruff>=0.9", "mypy>=1.14"]

[project.scripts]
newsninja = "newsninja.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
```

- [ ] **Step 3: Write the failing test**

`api/tests/conftest.py`:

```python
import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No test may read the developer's real credentials.

    Uses monkeypatch throughout so every change is undone after each test —
    setting os.environ directly would leak across tests.
    """
    for key in (
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "ENABLE_ORPHEUS",
        "CACHE_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
```

`api/tests/test_config.py`:

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from newsninja.config import Settings


def test_settings_reads_groq_key_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_example")
    settings = Settings(_env_file=None)
    assert settings.groq_api_key == "gsk_example"


def test_groq_key_is_required(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_reddit_credentials_default_to_none(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_example")
    settings = Settings(_env_file=None)
    assert settings.reddit_client_id is None
    assert settings.reddit_client_secret is None


def test_orpheus_is_off_by_default(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_example")
    assert Settings(_env_file=None).enable_orpheus is False


def test_cache_path_is_a_path(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_example")
    assert isinstance(Settings(_env_file=None).cache_path, Path)
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.config'`

- [ ] **Step 5: Write the implementation**

`api/newsninja/config.py`:

```python
"""Configuration loaded from environment or .env. Single source of truth."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Only ``groq_api_key`` is required. Reddit credentials are optional: without
    them the Reddit source reports itself unavailable rather than failing.
    """

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str

    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str = "newsninja/3.0 (portfolio project)"

    enable_orpheus: bool = False

    cache_path: Path = Path(".cache/newsninja.sqlite")
    request_timeout: int = 30


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Call this rather than constructing Settings."""
    return Settings()
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_config.py -v
```

Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
cd /Users/dhruv/Personal/NEWS-AI
git add -A
git commit -m "Replace legacy implementation with newsninja package scaffold

Deletes all twelve legacy source files per spec section 2: four competing
entry points, a services/ package that raised ImportError on import, and an
orphaned config.py. Adds the api/newsninja package with pydantic-settings
configuration and its tests."
```

---

### Task 2: Domain models and error types

**Files:**
- Create: `api/newsninja/models.py`, `api/newsninja/errors.py`, `api/tests/test_models.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Article(title, url, source, body, published)`, `Entity(name, kind)`, `Claim(text, quote)`, `ArticleAnalysis(topic, summary, entities, stance, confidence, key_claims)`, `Briefing(topics, language, script, analyses)`; errors `NewsNinjaError`, `SourceError`, `ExtractionFailure`, `RateLimitError`

- [ ] **Step 1: Write the failing test**

`api/tests/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from newsninja.models import Article, ArticleAnalysis, Briefing, Claim, Entity


def test_article_requires_title_and_url():
    article = Article(title="Apple rises", url="https://example.com/a", source="google_news")
    assert article.body == ""
    assert article.published is None


def test_entity_kind_is_constrained():
    with pytest.raises(ValidationError):
        Entity(name="Apple", kind="corporation")


def test_claim_carries_a_quote():
    claim = Claim(text="Apple stock rose", quote="Apple stock surged 8%")
    assert claim.quote == "Apple stock surged 8%"


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        ArticleAnalysis(
            topic="apple",
            summary="s",
            entities=[],
            stance="positive",
            confidence=1.5,
            key_claims=[],
        )


def test_stance_is_constrained():
    with pytest.raises(ValidationError):
        ArticleAnalysis(
            topic="apple",
            summary="s",
            entities=[],
            stance="Positive",  # capitalised — must fail
            confidence=0.5,
            key_claims=[],
        )


def test_briefing_defaults_to_english():
    briefing = Briefing(topics=["apple"], script="hello", analyses=[])
    assert briefing.language == "en"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.models'`

- [ ] **Step 3: Write the implementation**

`api/newsninja/errors.py`:

```python
"""Typed errors. Failures are surfaced, never swallowed into user-facing strings."""


class NewsNinjaError(Exception):
    """Base for every error this package raises."""


class SourceError(NewsNinjaError):
    """A source failed to fetch. Carries the source name so the UI can name it."""

    def __init__(self, source: str, message: str) -> None:
        self.source = source
        super().__init__(f"{source}: {message}")


class ExtractionFailure(NewsNinjaError):
    """The model could not produce schema-valid output within the retry budget."""


class RateLimitError(NewsNinjaError):
    """The provider's rate limit was hit. ``retry_after`` is in seconds."""

    def __init__(self, message: str, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(message)
```

`api/newsninja/models.py`:

```python
"""Domain models. These double as the LLM output contract via JSON Schema."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

EntityKind = Literal["person", "org", "place", "product", "other"]
Stance = Literal["positive", "negative", "neutral"]


class Article(BaseModel):
    """One fetched item. ``body`` is the text quote-grounding is checked against."""

    title: str
    url: str
    source: str
    body: str = ""
    published: datetime | None = None


class Entity(BaseModel):
    name: str
    kind: EntityKind


class Claim(BaseModel):
    """A factual assertion plus the verbatim span supporting it.

    ``quote`` must appear character-for-character in the source article. This is
    what makes hallucination checkable rather than a matter of opinion.
    """

    text: str
    quote: str


class ArticleAnalysis(BaseModel):
    topic: str
    summary: str
    entities: list[Entity]
    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    key_claims: list[Claim]


class Briefing(BaseModel):
    topics: list[str]
    script: str
    analyses: list[ArticleAnalysis]
    language: str = "en"
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_models.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add api/newsninja/models.py api/newsninja/errors.py api/tests/test_models.py
git commit -m "Add domain models and typed error hierarchy

Claim.quote carries the verbatim source span, which is what makes the
grounding metric in the eval harness a string containment check rather
than a judgement call."
```

---

### Task 3: Strict JSON Schema transform

Pydantic's `model_json_schema()` is rejected by Groq's strict mode. This task builds and tests the transform that fixes it. Verified against the live API on 2026-07-31: `additionalProperties: false` must be set on every object including those under `$defs`, and every property must be listed in `required`. `$ref`, `$defs`, `enum`, `minimum`, and `maximum` are all accepted unchanged.

**Files:**
- Create: `api/newsninja/analysis/__init__.py`, `api/newsninja/analysis/schema.py`, `api/tests/test_schema.py`

**Interfaces:**
- Consumes: `newsninja.models`
- Produces: `strict_schema(model: type[BaseModel]) -> dict[str, Any]`

- [ ] **Step 1: Write the failing test**

`api/tests/test_schema.py`:

```python
from newsninja.analysis.schema import strict_schema
from newsninja.models import ArticleAnalysis, Briefing


def _every_object(node):
    """Yield every JSON-Schema object node, including those under $defs."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _every_object(value)
    elif isinstance(node, list):
        for item in node:
            yield from _every_object(item)


def test_every_object_forbids_additional_properties():
    schema = strict_schema(ArticleAnalysis)
    objects = list(_every_object(schema))
    assert objects, "expected at least the root object"
    for obj in objects:
        assert obj["additionalProperties"] is False


def test_every_property_is_required():
    schema = strict_schema(ArticleAnalysis)
    for obj in _every_object(schema):
        assert set(obj["required"]) == set(obj["properties"])


def test_fields_with_defaults_become_required():
    """Pydantic omits defaulted fields from `required`; strict mode needs them."""
    schema = strict_schema(Briefing)
    assert "language" in schema["required"]


def test_defs_are_preserved():
    schema = strict_schema(ArticleAnalysis)
    assert "Entity" in schema["$defs"]
    assert schema["$defs"]["Entity"]["additionalProperties"] is False


def test_the_original_schema_is_not_mutated():
    original = ArticleAnalysis.model_json_schema()
    before = original.get("additionalProperties", "absent")
    strict_schema(ArticleAnalysis)
    assert original.get("additionalProperties", "absent") == before
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_schema.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.analysis'`

- [ ] **Step 3: Write the implementation**

```bash
touch api/newsninja/analysis/__init__.py
```

`api/newsninja/analysis/schema.py`:

```python
"""Convert Pydantic JSON Schema into the strict dialect Groq requires.

Groq rejects Pydantic's default output. Verified against the live API on
2026-07-31, strict mode requires:

  * ``additionalProperties: false`` on every object, including under ``$defs``
  * every property listed in ``required`` (Pydantic omits defaulted fields)

``$ref``, ``$defs``, ``enum``, ``minimum`` and ``maximum`` pass through unchanged.
"""

import copy
from typing import Any

from pydantic import BaseModel


def _tighten(node: Any) -> None:
    """Recursively apply the strict-dialect rules in place."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
            properties = node.get("properties", {})
            node["required"] = list(properties)
        for value in node.values():
            _tighten(value)
    elif isinstance(node, list):
        for item in node:
            _tighten(item)


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a strict-mode JSON Schema for ``model``.

    The caller's model is never mutated; the returned dict is a deep copy.
    """
    schema = copy.deepcopy(model.model_json_schema())
    _tighten(schema)
    return schema
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_schema.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add api/newsninja/analysis/ api/tests/test_schema.py
git commit -m "Add strict JSON Schema transform for Groq structured output

Pydantic's model_json_schema() is rejected by Groq strict mode: it omits
additionalProperties:false and leaves defaulted fields out of required.
Both requirements confirmed by live API request."
```

---

### Task 4: Token-budget rate limiter

The free tier is tokens-per-minute limited. This limiter waits *before* a call that would exceed the budget, rather than retrying into a 429. Clock and sleep are injected so tests never actually sleep.

**Files:**
- Create: `api/newsninja/analysis/limiter.py`, `api/tests/test_limiter.py`

**Interfaces:**
- Consumes: `newsninja.errors.RateLimitError`
- Produces: `TokenBudgetLimiter(tpm: int, clock=time.monotonic, sleeper=time.sleep)` with methods `reserve(estimated_tokens: int) -> None` and `observe(remaining: int, reset_seconds: float) -> None`

- [ ] **Step 1: Write the failing test**

`api/tests/test_limiter.py`:

```python
import pytest

from newsninja.analysis.limiter import TokenBudgetLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _limiter(tpm: int = 1000) -> tuple[TokenBudgetLimiter, FakeClock]:
    clock = FakeClock()
    return TokenBudgetLimiter(tpm=tpm, clock=clock.time, sleeper=clock.sleep), clock


def test_reserve_under_budget_does_not_sleep():
    limiter, clock = _limiter()
    limiter.reserve(400)
    assert clock.slept == []


def test_reserve_over_budget_sleeps_until_window_clears():
    limiter, clock = _limiter(tpm=1000)
    limiter.reserve(800)
    limiter.reserve(400)  # 1200 > 1000, must wait for the window to roll
    assert clock.slept, "expected the limiter to wait"


def test_window_rolls_after_sixty_seconds():
    limiter, clock = _limiter(tpm=1000)
    limiter.reserve(900)
    clock.now += 61
    limiter.reserve(900)
    assert clock.slept == [], "old usage should have aged out of the window"


def test_observe_low_remaining_forces_a_wait():
    limiter, clock = _limiter(tpm=1000)
    limiter.observe(remaining=10, reset_seconds=5.0)
    limiter.reserve(500)
    assert clock.slept == [5.0]


def test_reservation_larger_than_the_whole_budget_raises():
    limiter, _ = _limiter(tpm=1000)
    with pytest.raises(ValueError):
        limiter.reserve(1001)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_limiter.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.analysis.limiter'`

- [ ] **Step 3: Write the implementation**

`api/newsninja/analysis/limiter.py`:

```python
"""Sliding-window token budget limiter for Groq's free tier.

Waits before a call that would breach the budget instead of retrying into a 429.
The provider's own ``x-ratelimit-remaining-tokens`` header feeds back in via
``observe`` so the limiter corrects for usage it did not account for.
"""

import time
from collections import deque
from collections.abc import Callable

WINDOW_SECONDS = 60.0


class TokenBudgetLimiter:
    def __init__(
        self,
        tpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._tpm = tpm
        self._clock = clock
        self._sleep = sleeper
        self._events: deque[tuple[float, int]] = deque()
        self._forced_wait_until: float = 0.0

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
            self._events.popleft()

    def _used(self, now: float) -> int:
        self._prune(now)
        return sum(tokens for _, tokens in self._events)

    def reserve(self, estimated_tokens: int) -> None:
        """Block until ``estimated_tokens`` fits inside the budget, then record it."""
        if estimated_tokens > self._tpm:
            raise ValueError(
                f"reservation of {estimated_tokens} exceeds the entire "
                f"per-minute budget of {self._tpm}; split the request"
            )

        now = self._clock()
        if now < self._forced_wait_until:
            self._sleep(self._forced_wait_until - now)
            now = self._clock()

        while self._used(now) + estimated_tokens > self._tpm:
            oldest_at, _ = self._events[0]
            self._sleep(max(0.0, oldest_at + WINDOW_SECONDS - now))
            now = self._clock()

        self._events.append((now, estimated_tokens))

    def observe(self, remaining: int, reset_seconds: float) -> None:
        """Record the provider's own view of the budget.

        When the provider says very little is left, force a wait until its stated
        reset regardless of what the local window believes.
        """
        if remaining < self._tpm * 0.05:
            self._forced_wait_until = self._clock() + reset_seconds
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_limiter.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add api/newsninja/analysis/limiter.py api/tests/test_limiter.py
git commit -m "Add sliding-window token budget limiter

Groq's free tier is TPM-limited (8k on gpt-oss models, 12k on llama-3.3-70b).
Waits pre-emptively rather than retrying into 429s. Clock and sleep are
injected so tests run instantly."
```

---

### Task 5: Groq client with schema calls and validation-feedback retries

**Files:**
- Create: `api/newsninja/analysis/client.py`, `api/tests/test_client.py`

**Interfaces:**
- Consumes: `strict_schema`, `TokenBudgetLimiter`, `ExtractionFailure`, `RateLimitError`
- Produces: `GroqClient(api_key, limiters=None, transport=None)` with `structured(model, system, user, schema_model, max_retries=2) -> BaseModel`, `text(model, system, user) -> str`, and property `usage: Usage` where `Usage` has `prompt_tokens: int`, `completion_tokens: int`, `calls: int`

- [ ] **Step 1: Write the failing test**

`api/tests/test_client.py`:

```python
import pytest
from pydantic import BaseModel

from newsninja.analysis.client import GroqClient
from newsninja.errors import ExtractionFailure


class Tiny(BaseModel):
    name: str
    score: float


class StubTransport:
    """Stands in for the Groq SDK. Returns queued payloads in order."""

    def __init__(self, payloads: list[str]) -> None:
        self.payloads = list(payloads)
        self.requests: list[dict] = []

    def complete(self, **kwargs) -> tuple[str, dict[str, int]]:
        self.requests.append(kwargs)
        if not self.payloads:
            raise AssertionError("transport called more times than expected")
        return self.payloads.pop(0), {"prompt_tokens": 10, "completion_tokens": 5}


def test_structured_returns_a_validated_model():
    transport = StubTransport(['{"name": "apple", "score": 0.9}'])
    client = GroqClient(api_key="test", transport=transport)
    result = client.structured(
        model="openai/gpt-oss-20b", system="s", user="u", schema_model=Tiny
    )
    assert isinstance(result, Tiny)
    assert result.name == "apple"


def test_structured_sends_a_strict_schema():
    transport = StubTransport(['{"name": "a", "score": 0.1}'])
    client = GroqClient(api_key="test", transport=transport)
    client.structured(model="openai/gpt-oss-20b", system="s", user="u", schema_model=Tiny)

    schema = transport.requests[0]["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False


def test_invalid_output_is_retried_with_the_validation_error():
    transport = StubTransport(
        ['{"name": "a"}', '{"name": "a", "score": 0.4}']  # first is missing `score`
    )
    client = GroqClient(api_key="test", transport=transport)
    result = client.structured(
        model="openai/gpt-oss-20b", system="s", user="u", schema_model=Tiny
    )

    assert result.score == 0.4
    retry_messages = transport.requests[1]["messages"]
    assert any("score" in m["content"] for m in retry_messages), (
        "the retry must feed the validation error back to the model"
    )


def test_exhausting_retries_raises_extraction_failure():
    transport = StubTransport(['{"bad": 1}', '{"bad": 2}', '{"bad": 3}'])
    client = GroqClient(api_key="test", transport=transport)
    with pytest.raises(ExtractionFailure):
        client.structured(
            model="openai/gpt-oss-20b", system="s", user="u", schema_model=Tiny
        )


def test_structured_rejects_models_without_schema_support():
    client = GroqClient(api_key="test", transport=StubTransport([]))
    with pytest.raises(ValueError, match="does not support"):
        client.structured(
            model="llama-3.3-70b-versatile", system="s", user="u", schema_model=Tiny
        )


def test_usage_accumulates():
    transport = StubTransport(['{"name": "a", "score": 0.1}'])
    client = GroqClient(api_key="test", transport=transport)
    client.structured(model="openai/gpt-oss-20b", system="s", user="u", schema_model=Tiny)
    assert client.usage.calls == 1
    assert client.usage.prompt_tokens == 10
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_client.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.analysis.client'`

- [ ] **Step 3: Write the implementation**

`api/newsninja/analysis/client.py`:

```python
"""The only module in this package that talks to Groq.

Responsibilities: strict-schema calls, validation-feedback retries, token
budgeting, and usage accounting. Keeping the SDK behind this seam means a
provider swap or a test stub touches exactly one file.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from newsninja.analysis.limiter import TokenBudgetLimiter
from newsninja.analysis.schema import strict_schema
from newsninja.errors import ExtractionFailure

T = TypeVar("T", bound=BaseModel)

# Verified 2026-07-31: only these accept response_format=json_schema.
SCHEMA_CAPABLE_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b"})

# Verified free-tier tokens-per-minute ceilings.
MODEL_TPM = {
    "openai/gpt-oss-20b": 8_000,
    "openai/gpt-oss-120b": 8_000,
    "llama-3.3-70b-versatile": 12_000,
}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0


class Transport(Protocol):
    """Minimal seam over the Groq SDK so tests need no network."""

    def complete(self, **kwargs: Any) -> tuple[str, dict[str, int]]:
        """Return (content, usage_dict)."""


class GroqTransport:
    def __init__(self, api_key: str) -> None:
        from groq import Groq

        self._client = Groq(api_key=api_key)

    def complete(self, **kwargs: Any) -> tuple[str, dict[str, int]]:
        response = self._client.chat.completions.create(**kwargs)
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
        )


def _estimate_tokens(text: str) -> int:
    """Rough pre-call estimate. Four characters per token is close enough
    for budgeting; the limiter self-corrects from response headers."""
    return max(1, len(text) // 4)


class GroqClient:
    def __init__(
        self,
        api_key: str,
        limiters: dict[str, TokenBudgetLimiter] | None = None,
        transport: Transport | None = None,
    ) -> None:
        self._transport = transport or GroqTransport(api_key)
        self._limiters = limiters or {
            model: TokenBudgetLimiter(tpm) for model, tpm in MODEL_TPM.items()
        }
        self.usage = Usage()

    def _reserve(self, model: str, prompt: str) -> None:
        limiter = self._limiters.get(model)
        if limiter is not None:
            limiter.reserve(_estimate_tokens(prompt))

    def _record(self, usage: dict[str, int]) -> None:
        self.usage.calls += 1
        self.usage.prompt_tokens += usage.get("prompt_tokens", 0)
        self.usage.completion_tokens += usage.get("completion_tokens", 0)

    def structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema_model: type[T],
        max_retries: int = 2,
    ) -> T:
        """Call ``model`` and return a validated ``schema_model`` instance.

        On a validation failure the error text is fed back to the model as a
        correction message rather than blindly resampling.
        """
        if model not in SCHEMA_CAPABLE_MODELS:
            raise ValueError(
                f"model {model!r} does not support strict json_schema; "
                f"use one of {sorted(SCHEMA_CAPABLE_MODELS)}"
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_model.__name__.lower(),
                "strict": True,
                "schema": strict_schema(schema_model),
            },
        }

        last_error = ""
        for attempt in range(max_retries + 1):
            self._reserve(model, system + user)
            content, usage = self._transport.complete(
                model=model, messages=messages, response_format=response_format
            )
            self._record(usage)

            try:
                return schema_model.model_validate_json(content)
            except ValidationError as exc:
                last_error = str(exc)
                if attempt == max_retries:
                    break
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "That response failed schema validation with the "
                            f"following errors:\n{last_error}\n"
                            "Return corrected JSON matching the schema exactly."
                        ),
                    },
                ]

        raise ExtractionFailure(
            f"{schema_model.__name__} did not validate after "
            f"{max_retries + 1} attempts: {last_error}"
        )

    def text(self, *, model: str, system: str, user: str) -> str:
        """Free-form completion for prose that needs no schema."""
        self._reserve(model, system + user)
        content, usage = self._transport.complete(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        self._record(usage)
        return content
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_client.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add api/newsninja/analysis/client.py api/tests/test_client.py
git commit -m "Add Groq client with strict schema and validation-feedback retries

Retries feed the Pydantic error back to the model rather than resampling.
Rejects models that cannot do constrained decoding at call time instead of
failing opaquely at the API."
```

---

### Task 6: Source protocol and Google News source

**Files:**
- Create: `api/newsninja/sources/__init__.py`, `api/newsninja/sources/base.py`, `api/newsninja/sources/google_news.py`, `api/tests/fixtures/google_news_ai.xml`, `api/tests/test_sources_google_news.py`

**Interfaces:**
- Consumes: `Article`, `SourceError`
- Produces: `Source` protocol with `name: str`, `available() -> bool`, `fetch(topic, limit=8) -> list[Article]`; and `GoogleNewsSource(fetcher=None)`

- [ ] **Step 1: Record the fixture**

```bash
mkdir -p api/tests/fixtures
curl -s "https://news.google.com/rss/search?q=artificial+intelligence&hl=en-US&gl=US&ceid=US:en" \
  -o api/tests/fixtures/google_news_ai.xml
head -c 300 api/tests/fixtures/google_news_ai.xml
```

Expected: XML beginning `<?xml version="1.0"?><rss`. If the file is empty or an error page, stop and report — the rest of this task depends on it.

- [ ] **Step 2: Write the failing test**

`api/tests/test_sources_google_news.py`:

```python
from pathlib import Path

import pytest

from newsninja.errors import SourceError
from newsninja.sources.google_news import GoogleNewsSource

FIXTURE = Path(__file__).parent / "fixtures" / "google_news_ai.xml"


def _source() -> GoogleNewsSource:
    return GoogleNewsSource(fetcher=lambda url, timeout: FIXTURE.read_bytes())


def test_is_always_available():
    assert _source().available() is True


def test_fetch_returns_articles():
    articles = _source().fetch("artificial intelligence")
    assert articles
    assert all(a.source == "google_news" for a in articles)
    assert all(a.title for a in articles)
    assert all(a.url.startswith("http") for a in articles)


def test_fetch_respects_the_limit():
    assert len(_source().fetch("artificial intelligence", limit=3)) == 3


def test_titles_have_html_stripped():
    for article in _source().fetch("artificial intelligence"):
        assert "<" not in article.title


def test_transport_failure_raises_source_error():
    def boom(url, timeout):
        raise OSError("network down")

    with pytest.raises(SourceError) as excinfo:
        GoogleNewsSource(fetcher=boom).fetch("ai")
    assert excinfo.value.source == "google_news"
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_sources_google_news.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.sources'`

- [ ] **Step 4: Write the implementation**

```bash
touch api/newsninja/sources/__init__.py
```

`api/newsninja/sources/base.py`:

```python
"""The seam every source implements.

Adding a source is a new file, never an edit to the pipeline.
"""

from typing import Protocol, runtime_checkable

from newsninja.models import Article


@runtime_checkable
class Source(Protocol):
    name: str

    def available(self) -> bool:
        """False when required credentials are missing. The UI reports this
        as unavailable rather than as a failure."""

    def fetch(self, topic: str, limit: int = 8) -> list[Article]:
        """Fetch up to ``limit`` articles. Raises SourceError on transport failure."""
```

`api/newsninja/sources/google_news.py`:

```python
"""Google News RSS. Free, no authentication. Verified reachable 2026-07-31."""

import re
from collections.abc import Callable
from urllib.parse import quote_plus

import feedparser

from newsninja.errors import SourceError
from newsninja.models import Article

_TAG = re.compile(r"<[^>]+>")
_USER_AGENT = "Mozilla/5.0 (compatible; NewsNinja/3.0; +https://github.com/DHRUV1817)"


def _default_fetcher(url: str, timeout: int) -> bytes:
    import httpx

    response = httpx.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.content


class GoogleNewsSource:
    name = "google_news"

    def __init__(
        self,
        fetcher: Callable[[str, int], bytes] = _default_fetcher,
        timeout: int = 30,
    ) -> None:
        self._fetch_bytes = fetcher
        self._timeout = timeout

    def available(self) -> bool:
        return True

    def fetch(self, topic: str, limit: int = 8) -> list[Article]:
        url = (
            "https://news.google.com/rss/search"
            f"?q={quote_plus(topic)}&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            raw = self._fetch_bytes(url, self._timeout)
        except Exception as exc:
            raise SourceError(self.name, str(exc)) from exc

        feed = feedparser.parse(raw)
        articles: list[Article] = []
        for entry in feed.entries[:limit]:
            title = _TAG.sub("", getattr(entry, "title", "")).strip()
            if not title:
                continue
            summary = _TAG.sub("", getattr(entry, "summary", "")).strip()
            articles.append(
                Article(
                    title=title,
                    url=getattr(entry, "link", ""),
                    source=self.name,
                    body=f"{title}. {summary}".strip(),
                )
            )
        return articles
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_sources_google_news.py -v
```

Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add api/newsninja/sources/ api/tests/fixtures/google_news_ai.xml api/tests/test_sources_google_news.py
git commit -m "Add Source protocol and Google News RSS source

Transport is injected so tests run against a recorded fixture with no
network. Transport failures raise SourceError rather than returning a
user-facing string, which is what made the old code's failures
indistinguishable from empty results."
```

---

### Task 7: Reddit source via the official API

The public `search.json` endpoint returns 403 — verified from the developer's own machine on 2026-07-31, so the previous implementation is already broken. This uses the official OAuth API, and reports itself unavailable when credentials are absent.

**Files:**
- Create: `api/newsninja/sources/reddit.py`, `api/tests/test_sources_reddit.py`

**Interfaces:**
- Consumes: `Article`, `SourceError`, `Settings`
- Produces: `RedditSource(client_id, client_secret, user_agent, client_factory=None)`

- [ ] **Step 1: Write the failing test**

`api/tests/test_sources_reddit.py`:

```python
import pytest

from newsninja.errors import SourceError
from newsninja.sources.reddit import RedditSource


class FakeSubmission:
    def __init__(self, title, score, num_comments, permalink, selftext=""):
        self.title = title
        self.score = score
        self.num_comments = num_comments
        self.permalink = permalink
        self.selftext = selftext


class FakeReddit:
    def __init__(self, submissions):
        self._submissions = submissions

    def subreddit(self, name):
        return self

    def search(self, query, sort="hot", time_filter="week", limit=8):
        return iter(self._submissions[:limit])


def _source(submissions=None):
    submissions = submissions if submissions is not None else [
        FakeSubmission("AI breakthrough announced", 420, 87, "/r/tech/1"),
        FakeSubmission("Thoughts on the new model", 88, 12, "/r/tech/2"),
    ]
    return RedditSource(
        client_id="id",
        client_secret="secret",
        user_agent="test",
        client_factory=lambda **kw: FakeReddit(submissions),
    )


def test_unavailable_without_credentials():
    source = RedditSource(client_id=None, client_secret=None, user_agent="test")
    assert source.available() is False


def test_available_with_credentials():
    assert _source().available() is True


def test_fetch_without_credentials_raises():
    source = RedditSource(client_id=None, client_secret=None, user_agent="test")
    with pytest.raises(SourceError):
        source.fetch("ai")


def test_fetch_returns_articles_with_engagement_in_the_body():
    articles = _source().fetch("ai")
    assert len(articles) == 2
    assert articles[0].source == "reddit"
    assert "420" in articles[0].body
    assert articles[0].url.startswith("https://www.reddit.com")


def test_api_failure_raises_source_error():
    class Exploding:
        def subreddit(self, name):
            raise RuntimeError("401 unauthorized")

    source = RedditSource(
        client_id="id",
        client_secret="secret",
        user_agent="test",
        client_factory=lambda **kw: Exploding(),
    )
    with pytest.raises(SourceError) as excinfo:
        source.fetch("ai")
    assert excinfo.value.source == "reddit"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_sources_reddit.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.sources.reddit'`

- [ ] **Step 3: Write the implementation**

`api/newsninja/sources/reddit.py`:

```python
"""Reddit via the official OAuth API.

The public search.json endpoint returns 403 (verified 2026-07-31), so a
script app at reddit.com/prefs/apps is required. Without credentials this
source reports itself unavailable rather than failing at call time.
"""

from collections.abc import Callable
from typing import Any

from newsninja.errors import SourceError
from newsninja.models import Article


def _default_client_factory(**kwargs: Any) -> Any:
    import praw

    return praw.Reddit(**kwargs)


class RedditSource:
    name = "reddit"

    def __init__(
        self,
        client_id: str | None,
        client_secret: str | None,
        user_agent: str,
        client_factory: Callable[..., Any] = _default_client_factory,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._factory = client_factory

    def available(self) -> bool:
        return bool(self._client_id and self._client_secret)

    def fetch(self, topic: str, limit: int = 8) -> list[Article]:
        if not self.available():
            raise SourceError(
                self.name,
                "credentials missing; create a script app at reddit.com/prefs/apps "
                "and set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET",
            )

        try:
            reddit = self._factory(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
            submissions = list(
                reddit.subreddit("all").search(
                    topic, sort="hot", time_filter="week", limit=limit
                )
            )
        except Exception as exc:
            raise SourceError(self.name, str(exc)) from exc

        articles: list[Article] = []
        for submission in submissions:
            body = (
                f"{submission.title}. Reddit discussion with {submission.score} "
                f"upvotes and {submission.num_comments} comments. {submission.selftext}"
            ).strip()
            articles.append(
                Article(
                    title=submission.title,
                    url=f"https://www.reddit.com{submission.permalink}",
                    source=self.name,
                    body=body,
                )
            )
        return articles
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_sources_reddit.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add api/newsninja/sources/reddit.py api/tests/test_sources_reddit.py
git commit -m "Add Reddit source via official OAuth API

The public search.json endpoint returns 403, so the previous scraper was
already broken. Credentials are optional: without them the source reports
unavailable rather than failing at call time."
```

---

### Task 8: SQLite cache

Without this the eval suite would exhaust the free tier on every CI run.

**Files:**
- Create: `api/newsninja/cache.py`, `api/tests/test_cache.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Cache(path: Path)` with `get(key: str) -> str | None`, `set(key: str, value: str) -> None`, `make_key(**parts) -> str`, `clear() -> None`

- [ ] **Step 1: Write the failing test**

`api/tests/test_cache.py`:

```python
from newsninja.cache import Cache


def test_get_returns_none_for_a_missing_key(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    assert cache.get("nope") is None


def test_set_then_get_round_trips(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    cache.set("k", '{"a": 1}')
    assert cache.get("k") == '{"a": 1}'


def test_survives_reopening(tmp_path):
    path = tmp_path / "c.sqlite"
    Cache(path).set("k", "v")
    assert Cache(path).get("k") == "v"


def test_make_key_is_stable_regardless_of_argument_order():
    a = Cache.make_key(topic="ai", model="m", prompt_version="1")
    b = Cache.make_key(prompt_version="1", model="m", topic="ai")
    assert a == b


def test_make_key_changes_when_prompt_version_changes():
    a = Cache.make_key(topic="ai", model="m", prompt_version="1")
    b = Cache.make_key(topic="ai", model="m", prompt_version="2")
    assert a != b


def test_clear_empties_the_cache(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    cache.set("k", "v")
    cache.clear()
    assert cache.get("k") is None


def test_creates_parent_directories(tmp_path):
    cache = Cache(tmp_path / "nested" / "deeper" / "c.sqlite")
    cache.set("k", "v")
    assert cache.get("k") == "v"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_cache.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.cache'`

- [ ] **Step 3: Write the implementation**

`api/newsninja/cache.py`:

```python
"""SQLite response cache.

Keyed on the full call identity — topic, source, model, and prompt version —
so changing a prompt invalidates exactly the affected entries. Without this the
eval suite would exhaust the free tier on every CI run.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
)
"""


class Cache:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    @staticmethod
    def make_key(**parts: str) -> str:
        """Order-independent cache key over the full call identity."""
        canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM entries WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO entries (key, value) VALUES (?, ?)",
                (key, value),
            )

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM entries")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_cache.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add api/newsninja/cache.py api/tests/test_cache.py
git commit -m "Add SQLite cache keyed on full call identity

Key includes prompt_version so editing a prompt invalidates exactly the
affected entries and evals can compare versions."
```

---

### Task 9: Extraction

Batches all articles for one topic into a single structured call — five requests instead of forty, which is what keeps the run inside the 8,000 TPM ceiling.

**Files:**
- Create: `api/newsninja/analysis/prompts/__init__.py`, `api/newsninja/analysis/prompts/extract.md`, `api/newsninja/analysis/extract.py`, `api/tests/test_extract.py`

**Interfaces:**
- Consumes: `GroqClient`, `Cache`, `Article`, `ArticleAnalysis`
- Produces: `PROMPT_VERSION: str`, `EXTRACT_SYSTEM: str`, `extract_topic(client, topic, articles, cache=None, model="openai/gpt-oss-20b") -> ArticleAnalysis`

- [ ] **Step 1: Write the failing test**

`api/tests/test_extract.py`:

```python
import json

from newsninja.analysis.extract import extract_topic
from newsninja.cache import Cache
from newsninja.models import Article, ArticleAnalysis

VALID = {
    "topic": "ai",
    "summary": "Models improved.",
    "entities": [{"name": "OpenAI", "kind": "org"}],
    "stance": "positive",
    "confidence": 0.8,
    "key_claims": [{"text": "Models improved", "quote": "models improved this year"}],
}


class RecordingClient:
    def __init__(self, payload=None):
        self.payload = payload or VALID
        self.calls = []

    def structured(self, *, model, system, user, schema_model, max_retries=2):
        self.calls.append({"model": model, "system": system, "user": user})
        return schema_model.model_validate(self.payload)


def _articles():
    return [
        Article(title="A", url="u1", source="google_news", body="models improved this year"),
        Article(title="B", url="u2", source="reddit", body="people are excited"),
    ]


def test_returns_an_analysis():
    result = extract_topic(RecordingClient(), "ai", _articles())
    assert isinstance(result, ArticleAnalysis)
    assert result.stance == "positive"


def test_all_articles_go_in_one_call():
    client = RecordingClient()
    extract_topic(client, "ai", _articles())
    assert len(client.calls) == 1, "articles must be batched into a single request"
    assert "models improved this year" in client.calls[0]["user"]
    assert "people are excited" in client.calls[0]["user"]


def test_uses_a_schema_capable_model():
    client = RecordingClient()
    extract_topic(client, "ai", _articles())
    assert client.calls[0]["model"] == "openai/gpt-oss-20b"


def test_empty_articles_returns_a_neutral_analysis_without_calling_the_model():
    client = RecordingClient()
    result = extract_topic(client, "ai", [])
    assert client.calls == []
    assert result.stance == "neutral"
    assert result.key_claims == []


def test_result_is_cached(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    client = RecordingClient()
    extract_topic(client, "ai", _articles(), cache=cache)
    extract_topic(client, "ai", _articles(), cache=cache)
    assert len(client.calls) == 1, "second call must be served from cache"


def test_cached_value_round_trips(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    first = extract_topic(RecordingClient(), "ai", _articles(), cache=cache)
    second = extract_topic(RecordingClient(), "ai", _articles(), cache=cache)
    assert first == second
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_extract.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.analysis.extract'`

- [ ] **Step 3: Write the prompt**

```bash
mkdir -p api/newsninja/analysis/prompts
```

`api/newsninja/analysis/prompts/extract.md`:

```markdown
You are a news analyst. You are given several articles about one topic.

Produce a single consolidated analysis of that topic:

- `summary`: two to four sentences covering what the articles collectively report.
  Neutral register. No editorialising.
- `entities`: the named people, organisations, places and products that actually
  appear in the articles. Do not infer entities that are not named.
- `stance`: the overall tone the coverage takes toward the topic.
- `confidence`: how confident you are in the stance, from 0.0 to 1.0. Use low values
  when the coverage is mixed or thin.
- `key_claims`: the most important factual assertions. Each claim MUST include a
  `quote` copied character-for-character from the supplied article text. If you
  cannot find an exact supporting span, omit the claim entirely.

Never invent a quote. A claim without a verbatim source span is worse than no claim.
```

`api/newsninja/analysis/prompts/__init__.py`:

```python
"""Versioned prompts. PROMPT_VERSION participates in the cache key, so editing
a prompt invalidates exactly the entries it affects."""

from pathlib import Path

PROMPT_VERSION = "1"

_DIR = Path(__file__).parent


def load(name: str) -> str:
    return (_DIR / f"{name}.md").read_text(encoding="utf-8")


EXTRACT_SYSTEM = load("extract")
```

- [ ] **Step 4: Write the implementation**

`api/newsninja/analysis/extract.py`:

```python
"""Topic extraction.

All articles for one topic go into a single structured call. At eight articles
per topic and five topics, per-article calls would be forty requests and roughly
32,000 tokens — well past the 8,000 TPM free-tier ceiling. Batching makes it five.
"""

from newsninja.analysis.prompts import EXTRACT_SYSTEM, PROMPT_VERSION
from newsninja.cache import Cache
from newsninja.models import Article, ArticleAnalysis

DEFAULT_MODEL = "openai/gpt-oss-20b"


def _render(topic: str, articles: list[Article]) -> str:
    lines = [f"Topic: {topic}", "", "Articles:"]
    for index, article in enumerate(articles, start=1):
        lines.append(f"[{index}] ({article.source}) {article.title}")
        lines.append(article.body)
        lines.append("")
    return "\n".join(lines)


def _empty(topic: str) -> ArticleAnalysis:
    return ArticleAnalysis(
        topic=topic,
        summary=f"No articles were found for {topic}.",
        entities=[],
        stance="neutral",
        confidence=0.0,
        key_claims=[],
    )


def extract_topic(
    client,
    topic: str,
    articles: list[Article],
    cache: Cache | None = None,
    model: str = DEFAULT_MODEL,
) -> ArticleAnalysis:
    """Consolidate ``articles`` into one typed analysis of ``topic``."""
    if not articles:
        return _empty(topic)

    user = _render(topic, articles)

    key = None
    if cache is not None:
        key = Cache.make_key(
            kind="extract",
            topic=topic,
            model=model,
            prompt_version=PROMPT_VERSION,
            payload=user,
        )
        cached = cache.get(key)
        if cached is not None:
            return ArticleAnalysis.model_validate_json(cached)

    analysis = client.structured(
        model=model,
        system=EXTRACT_SYSTEM,
        user=user,
        schema_model=ArticleAnalysis,
    )

    if cache is not None and key is not None:
        cache.set(key, analysis.model_dump_json())

    return analysis
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_extract.py -v
```

Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add api/newsninja/analysis/prompts/ api/newsninja/analysis/extract.py api/tests/test_extract.py
git commit -m "Add batched topic extraction with versioned prompts

One structured call per topic rather than one per article: five requests
instead of forty, which is what keeps a run inside the 8k TPM ceiling."
```

---

### Task 10: Briefing synthesis and translation

**Files:**
- Create: `api/newsninja/analysis/prompts/synthesize.md`, `api/newsninja/analysis/prompts/translate.md`, `api/newsninja/analysis/synthesize.py`, `api/tests/test_synthesize.py`
- Modify: `api/newsninja/analysis/prompts/__init__.py`

**Interfaces:**
- Consumes: `GroqClient`, `ArticleAnalysis`, `Briefing`
- Produces: `synthesize(client, analyses, model="openai/gpt-oss-120b") -> str`, `translate(client, script, language, model="llama-3.3-70b-versatile") -> str`, `build_briefing(client, analyses, language="en") -> Briefing`

- [ ] **Step 1: Write the failing test**

`api/tests/test_synthesize.py`:

```python
from newsninja.analysis.synthesize import build_briefing, synthesize, translate
from newsninja.models import ArticleAnalysis, Briefing, Claim, Entity


def _analysis(topic="ai", stance="positive"):
    return ArticleAnalysis(
        topic=topic,
        summary=f"Summary about {topic}.",
        entities=[Entity(name="OpenAI", kind="org")],
        stance=stance,
        confidence=0.8,
        key_claims=[Claim(text="c", quote="q")],
    )


class StubClient:
    def __init__(self, reply="Good evening. Here is your briefing."):
        self.reply = reply
        self.calls = []

    def text(self, *, model, system, user):
        self.calls.append({"model": model, "system": system, "user": user})
        return self.reply

    def structured(self, **kwargs):
        raise AssertionError("synthesis must use free-form text, not a schema call")


def test_synthesize_returns_a_script():
    client = StubClient()
    script = synthesize(client, [_analysis()])
    assert script == "Good evening. Here is your briefing."


def test_synthesize_uses_the_reasoning_model():
    client = StubClient()
    synthesize(client, [_analysis()])
    assert client.calls[0]["model"] == "openai/gpt-oss-120b"


def test_synthesize_includes_every_topic():
    client = StubClient()
    synthesize(client, [_analysis("ai"), _analysis("climate")])
    user = client.calls[0]["user"]
    assert "ai" in user and "climate" in user


def test_translate_uses_the_high_budget_model():
    client = StubClient(reply="Bonjour.")
    translate(client, "Hello.", "fr")
    assert client.calls[0]["model"] == "llama-3.3-70b-versatile"
    assert "fr" in client.calls[0]["user"] or "fr" in client.calls[0]["system"]


def test_translate_to_english_is_a_no_op():
    client = StubClient()
    assert translate(client, "Hello.", "en") == "Hello."
    assert client.calls == [], "English needs no translation call"


def test_build_briefing_returns_a_briefing():
    briefing = build_briefing(StubClient(), [_analysis()])
    assert isinstance(briefing, Briefing)
    assert briefing.topics == ["ai"]
    assert briefing.language == "en"


def test_build_briefing_translates_when_asked():
    client = StubClient(reply="translated")
    briefing = build_briefing(client, [_analysis()], language="fr")
    assert briefing.language == "fr"
    assert briefing.script == "translated"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_synthesize.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.analysis.synthesize'`

- [ ] **Step 3: Write the prompts**

`api/newsninja/analysis/prompts/synthesize.md`:

```markdown
You write short broadcast news briefings.

Given per-topic analyses, write a spoken-word script that covers each topic in turn.

Rules:

- Open with one orienting sentence. Close with one sentence. Nothing ceremonial.
- Cover every topic supplied, in the order given.
- Report what the analyses say. Do not add facts, figures, or context of your own.
- Where an analysis reports low confidence, say the coverage is mixed or thin
  rather than asserting a conclusion.
- Plain spoken prose. No headings, no bullet points, no stage directions — this
  text goes straight to a speech synthesiser.
```

`api/newsninja/analysis/prompts/translate.md`:

```markdown
You are a translator. Translate the supplied broadcast script into the requested
language.

Preserve meaning, register, and sentence rhythm. The result will be read aloud by
a speech synthesiser, so it must be natural spoken prose in the target language.

Return only the translation. No notes, no preamble, no source text.
```

Modify `api/newsninja/analysis/prompts/__init__.py` — append after `EXTRACT_SYSTEM`:

```python
SYNTHESIZE_SYSTEM = load("synthesize")
TRANSLATE_SYSTEM = load("translate")
```

- [ ] **Step 4: Write the implementation**

`api/newsninja/analysis/synthesize.py`:

```python
"""Briefing synthesis and translation.

Both produce prose rather than structured data, so they use free-form calls and
can run on llama-3.3-70b-versatile, which carries the highest free-tier token
budget (12,000 TPM). Synthesis uses gpt-oss-120b for its stronger reasoning.
"""

from newsninja.analysis.prompts import SYNTHESIZE_SYSTEM, TRANSLATE_SYSTEM
from newsninja.models import ArticleAnalysis, Briefing

SYNTHESIS_MODEL = "openai/gpt-oss-120b"
TRANSLATION_MODEL = "llama-3.3-70b-versatile"


def _render(analyses: list[ArticleAnalysis]) -> str:
    blocks = []
    for analysis in analyses:
        entities = ", ".join(entity.name for entity in analysis.entities) or "none"
        claims = "\n".join(f"  - {claim.text}" for claim in analysis.key_claims) or "  - none"
        blocks.append(
            f"Topic: {analysis.topic}\n"
            f"Summary: {analysis.summary}\n"
            f"Stance: {analysis.stance} (confidence {analysis.confidence:.2f})\n"
            f"Entities: {entities}\n"
            f"Key claims:\n{claims}"
        )
    return "\n\n".join(blocks)


def synthesize(client, analyses: list[ArticleAnalysis], model: str = SYNTHESIS_MODEL) -> str:
    """Turn per-topic analyses into a spoken-word briefing script."""
    return client.text(model=model, system=SYNTHESIZE_SYSTEM, user=_render(analyses))


def translate(
    client, script: str, language: str, model: str = TRANSLATION_MODEL
) -> str:
    """Translate ``script`` into ``language``. English is a no-op."""
    if language == "en":
        return script
    return client.text(
        model=model,
        system=TRANSLATE_SYSTEM,
        user=f"Target language code: {language}\n\nScript:\n{script}",
    )


def build_briefing(
    client, analyses: list[ArticleAnalysis], language: str = "en"
) -> Briefing:
    """Synthesise, translate if needed, and package the result."""
    script = synthesize(client, analyses)
    script = translate(client, script, language)
    return Briefing(
        topics=[analysis.topic for analysis in analyses],
        script=script,
        analyses=analyses,
        language=language,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_synthesize.py -v
```

Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add api/newsninja/analysis/prompts/ api/newsninja/analysis/synthesize.py api/tests/test_synthesize.py
git commit -m "Add briefing synthesis and translation

Multi-language now translates the briefing itself rather than only passing a
language code to the speech synthesiser, which is what the old code did while
leaving the text in English."
```

---

### Task 11: Text to speech with language routing

Orpheus covers English and Saudi Arabic only, with a 4,000-token context — verified 2026-07-31. It also returns `400 model_terms_required` until terms are accepted at console.groq.com. gTTS is therefore the default and Orpheus an opt-in upgrade that degrades gracefully.

**Files:**
- Create: `api/newsninja/audio/__init__.py`, `api/newsninja/audio/tts.py`, `api/tests/test_tts.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `ORPHEUS_LANGUAGES: frozenset[str]`, `SUPPORTED_LANGUAGES: tuple[str, ...]`, `chunk_text(text, limit=3000) -> list[str]`, `synthesize_speech(text, language="en", enable_orpheus=False, gtts_factory=None, orpheus_fn=None) -> bytes`

- [ ] **Step 1: Write the failing test**

`api/tests/test_tts.py`:

```python
import pytest

from newsninja.audio.tts import (
    ORPHEUS_LANGUAGES,
    SUPPORTED_LANGUAGES,
    chunk_text,
    synthesize_speech,
)


class FakeGTTS:
    instances = []

    def __init__(self, text, lang, slow=False):
        self.text = text
        self.lang = lang
        FakeGTTS.instances.append(self)

    def write_to_fp(self, fp):
        fp.write(b"MP3" + self.text.encode()[:4])


@pytest.fixture(autouse=True)
def _reset():
    FakeGTTS.instances = []


def test_chunk_text_keeps_short_text_whole():
    assert chunk_text("hello there") == ["hello there"]


def test_chunk_text_splits_on_sentence_boundaries():
    text = ". ".join(f"sentence number {i}" for i in range(200)) + "."
    chunks = chunk_text(text, limit=200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_loses_no_words():
    text = ". ".join(f"sentence number {i}" for i in range(50)) + "."
    joined = " ".join(chunk_text(text, limit=120))
    assert joined.replace("  ", " ").split() == text.split()


def test_gtts_is_used_by_default():
    audio = synthesize_speech("hello", language="en", gtts_factory=FakeGTTS)
    assert audio.startswith(b"MP3")
    assert FakeGTTS.instances[0].lang == "en"


def test_unsupported_language_falls_back_to_english():
    synthesize_speech("hello", language="xx", gtts_factory=FakeGTTS)
    assert FakeGTTS.instances[0].lang == "en"


def test_orpheus_used_for_english_when_enabled():
    calls = []

    def orpheus(text, language):
        calls.append(language)
        return b"WAVdata"

    audio = synthesize_speech(
        "hello", language="en", enable_orpheus=True,
        gtts_factory=FakeGTTS, orpheus_fn=orpheus,
    )
    assert audio == b"WAVdata"
    assert calls == ["en"]


def test_orpheus_skipped_for_unsupported_language():
    def orpheus(text, language):
        raise AssertionError("Orpheus must not be called for French")

    synthesize_speech(
        "bonjour", language="fr", enable_orpheus=True,
        gtts_factory=FakeGTTS, orpheus_fn=orpheus,
    )
    assert FakeGTTS.instances[0].lang == "fr"


def test_orpheus_failure_falls_back_to_gtts():
    def orpheus(text, language):
        raise RuntimeError("400 model_terms_required")

    audio = synthesize_speech(
        "hello", language="en", enable_orpheus=True,
        gtts_factory=FakeGTTS, orpheus_fn=orpheus,
    )
    assert audio.startswith(b"MP3")


def test_orpheus_language_set_is_english_and_arabic():
    assert ORPHEUS_LANGUAGES == frozenset({"en", "ar"})


def test_twelve_languages_are_supported():
    assert len(SUPPORTED_LANGUAGES) == 12
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_tts.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.audio'`

- [ ] **Step 3: Write the implementation**

```bash
mkdir -p api/newsninja/audio && touch api/newsninja/audio/__init__.py
```

`api/newsninja/audio/tts.py`:

```python
"""Speech synthesis with honest language routing.

gTTS is the default: free, no authentication, twelve languages. Orpheus is an
opt-in upgrade that produces markedly better audio but covers only English and
Saudi Arabic (verified 2026-07-31) and returns 400 until its terms are accepted
at console.groq.com. Any Orpheus failure falls back to gTTS rather than
surfacing as a broken feature.
"""

import io
import re
from collections.abc import Callable

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "hi", "ar",
)

ORPHEUS_LANGUAGES = frozenset({"en", "ar"})

ORPHEUS_MODELS = {
    "en": "canopylabs/orpheus-v1-english",
    "ar": "canopylabs/orpheus-arabic-saudi",
}

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, limit: int = 3000) -> list[str]:
    """Split on sentence boundaries into chunks of at most ``limit`` characters.

    Orpheus has a 4,000-token context, and gTTS degrades on very long input.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE.split(text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _gtts_bytes(text: str, language: str, factory: Callable) -> bytes:
    buffer = io.BytesIO()
    for chunk in chunk_text(text):
        factory(text=chunk, lang=language, slow=False).write_to_fp(buffer)
    return buffer.getvalue()


def _default_gtts_factory(**kwargs):
    from gtts import gTTS

    return gTTS(**kwargs)


def synthesize_speech(
    text: str,
    language: str = "en",
    enable_orpheus: bool = False,
    gtts_factory: Callable | None = None,
    orpheus_fn: Callable[[str, str], bytes] | None = None,
) -> bytes:
    """Render ``text`` to audio bytes, routing by language and availability."""
    if language not in SUPPORTED_LANGUAGES:
        language = "en"

    factory = gtts_factory or _default_gtts_factory

    if enable_orpheus and language in ORPHEUS_LANGUAGES and orpheus_fn is not None:
        try:
            return orpheus_fn(text, language)
        except Exception:
            # Terms not accepted, or the model is unavailable. gTTS still works.
            pass

    return _gtts_bytes(text, language, factory)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_tts.py -v
```

Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add api/newsninja/audio/ api/tests/test_tts.py
git commit -m "Add TTS with honest language routing

Orpheus covers English and Saudi Arabic only and needs terms acceptance, so
gTTS is the default and Orpheus an opt-in upgrade that falls back rather than
breaking. The README must state this split rather than implying uniform
neural quality."
```

---

### Task 12: Pipeline orchestration

**Files:**
- Create: `api/newsninja/pipeline.py`, `api/tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above
- Produces: `PipelineResult(briefing, audio, source_errors)`, `run_pipeline(topics, sources, client, cache=None, language="en", enable_orpheus=False, limit=8, tts=None) -> PipelineResult`

- [ ] **Step 1: Write the failing test**

`api/tests/test_pipeline.py`:

```python
import pytest

from newsninja.errors import SourceError
from newsninja.models import Article, ArticleAnalysis
from newsninja.pipeline import run_pipeline


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


def _article(source="google_news"):
    return Article(title="T", url="u", source=source, body="b")


def _tts(text, language, enable_orpheus):
    return b"AUDIO"


def test_produces_a_briefing_and_audio():
    result = run_pipeline(
        topics=["ai"],
        sources=[FakeSource("google_news", [_article()])],
        client=FakeClient(),
        tts=_tts,
    )
    assert result.briefing.script == "Here is your briefing."
    assert result.audio == b"AUDIO"
    assert result.source_errors == {}


def test_a_failing_source_does_not_stop_the_others():
    result = run_pipeline(
        topics=["ai"],
        sources=[
            FakeSource("reddit", error=SourceError("reddit", "401")),
            FakeSource("google_news", [_article()]),
        ],
        client=FakeClient(),
        tts=_tts,
    )
    assert "reddit" in result.source_errors
    assert result.briefing.analyses


def test_unavailable_sources_are_skipped_without_error():
    result = run_pipeline(
        topics=["ai"],
        sources=[
            FakeSource("reddit", is_available=False),
            FakeSource("google_news", [_article()]),
        ],
        client=FakeClient(),
        tts=_tts,
    )
    assert result.source_errors == {}


def test_empty_topics_raises():
    with pytest.raises(ValueError):
        run_pipeline(topics=[], sources=[], client=FakeClient(), tts=_tts)


def test_more_than_five_topics_raises():
    with pytest.raises(ValueError, match="at most 5"):
        run_pipeline(
            topics=["a", "b", "c", "d", "e", "f"],
            sources=[FakeSource("google_news", [_article()])],
            client=FakeClient(),
            tts=_tts,
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_pipeline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.pipeline'`

- [ ] **Step 3: Write the implementation**

`api/newsninja/pipeline.py`:

```python
"""Orchestration: topics in, briefing and audio out.

Source failures are collected per source rather than aborting the run, so one
dead source degrades the briefing instead of destroying it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from newsninja.analysis.extract import extract_topic
from newsninja.analysis.synthesize import build_briefing
from newsninja.audio.tts import synthesize_speech
from newsninja.cache import Cache
from newsninja.errors import SourceError
from newsninja.models import Article, Briefing

MAX_TOPICS = 5


@dataclass
class PipelineResult:
    briefing: Briefing
    audio: bytes
    source_errors: dict[str, str] = field(default_factory=dict)


def _default_tts(text: str, language: str, enable_orpheus: bool) -> bytes:
    return synthesize_speech(text, language=language, enable_orpheus=enable_orpheus)


def run_pipeline(
    topics: list[str],
    sources: list,
    client,
    cache: Cache | None = None,
    language: str = "en",
    enable_orpheus: bool = False,
    limit: int = 8,
    tts: Callable[[str, str, bool], bytes] = _default_tts,
) -> PipelineResult:
    """Fetch, extract, synthesise, and render audio for ``topics``."""
    if not topics:
        raise ValueError("at least one topic is required")
    if len(topics) > MAX_TOPICS:
        raise ValueError(f"at most {MAX_TOPICS} topics per run, got {len(topics)}")

    source_errors: dict[str, str] = {}
    analyses = []

    for topic in topics:
        articles: list[Article] = []
        for source in sources:
            if not source.available():
                continue
            try:
                articles.extend(source.fetch(topic, limit=limit))
            except SourceError as exc:
                source_errors[source.name] = str(exc)
        analyses.append(extract_topic(client, topic, articles, cache=cache))

    briefing = build_briefing(client, analyses, language=language)
    audio = tts(briefing.script, language, enable_orpheus)

    return PipelineResult(briefing=briefing, audio=audio, source_errors=source_errors)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_pipeline.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add api/newsninja/pipeline.py api/tests/test_pipeline.py
git commit -m "Add pipeline orchestration

Source failures are collected per source rather than aborting the run, so one
dead source degrades the briefing instead of destroying it."
```

---

### Task 13: CLI, CI, and package README

Makes the package runnable end to end and provable in CI.

**Files:**
- Create: `api/newsninja/cli.py`, `api/README.md`, `.github/workflows/ci.yml`
- Modify: `api/tests/conftest.py`

**Interfaces:**
- Consumes: everything above
- Produces: `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

Append to `api/tests/test_pipeline.py`:

```python
def test_cli_reports_missing_credentials_without_traceback(capsys, monkeypatch):
    """The CLI must fail with a readable message, not a traceback.

    `Settings` normally reads ../.env, which on a developer machine holds a real
    key — so the class is swapped for one with env_file disabled. `cli.main`
    imports Settings inside the function body, so patching the module attribute
    takes effect at call time.
    """
    from pydantic_settings import SettingsConfigDict

    import newsninja.config as config_module
    from newsninja.cli import main

    class NoEnvFileSettings(config_module.Settings):
        model_config = SettingsConfigDict(env_file=None, extra="ignore")

    monkeypatch.setattr(config_module, "Settings", NoEnvFileSettings)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    exit_code = main(["--topic", "ai", "--no-audio"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "GROQ_API_KEY" in captured.err
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_pipeline.py::test_cli_reports_missing_credentials_without_traceback -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'newsninja.cli'`

- [ ] **Step 3: Write the implementation**

`api/newsninja/cli.py`:

```python
"""Command-line entry point.

Makes the whole pipeline runnable without a web server, which is what lets the
eval harness and CI exercise it directly.
"""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newsninja", description="Generate a source-grounded news briefing."
    )
    parser.add_argument("--topic", action="append", required=True,
                        help="Topic to analyse. Repeat for up to 5.")
    parser.add_argument("--language", default="en", help="Output language code.")
    parser.add_argument("--out", type=Path, default=Path("briefing.mp3"),
                        help="Where to write the audio.")
    parser.add_argument("--no-audio", action="store_true",
                        help="Print the script only; skip speech synthesis.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the cache.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from newsninja.config import Settings

    try:
        settings = Settings()
    except ValidationError:
        print(
            "GROQ_API_KEY is not set. Add it to .env or export it, then retry.",
            file=sys.stderr,
        )
        return 1

    from newsninja.analysis.client import GroqClient
    from newsninja.cache import Cache
    from newsninja.pipeline import run_pipeline
    from newsninja.sources.google_news import GoogleNewsSource
    from newsninja.sources.reddit import RedditSource

    sources = [
        GoogleNewsSource(timeout=settings.request_timeout),
        RedditSource(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        ),
    ]

    # `tts` must be omitted rather than passed as None, or it would override
    # run_pipeline's default with None and crash on call.
    pipeline_kwargs = {
        "topics": args.topic,
        "sources": sources,
        "client": GroqClient(api_key=settings.groq_api_key),
        "cache": None if args.no_cache else Cache(settings.cache_path),
        "language": args.language,
        "enable_orpheus": settings.enable_orpheus,
    }
    if args.no_audio:
        pipeline_kwargs["tts"] = lambda text, lang, orpheus: b""

    result = run_pipeline(**pipeline_kwargs)

    print(result.briefing.script)

    for name, message in result.source_errors.items():
        print(f"warning: source {name} failed: {message}", file=sys.stderr)

    if not args.no_audio:
        args.out.write_bytes(result.audio)
        print(f"\nAudio written to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full test suite**

```bash
cd api && uv run --python 3.12 --extra dev pytest -v
```

Expected: all tests pass (roughly 62 across 11 files).

- [ ] **Step 5: Add CI**

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: api
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5

      - name: Lint
        run: uv run --python 3.12 --extra dev ruff check .

      - name: Type check
        run: uv run --python 3.12 --extra dev mypy newsninja

      - name: Test
        run: uv run --python 3.12 --extra dev pytest --cov=newsninja --cov-report=term-missing
        env:
          GROQ_API_KEY: test-key-not-real
```

- [ ] **Step 6: Verify lint and types pass locally**

```bash
cd api && uv run --python 3.12 --extra dev ruff check . && uv run --python 3.12 --extra dev mypy newsninja
```

Expected: both clean. Fix any findings before committing.

- [ ] **Step 7: Write the package README**

`api/README.md`:

```markdown
# newsninja (core package)

Turns a list of topics into a source-grounded news briefing with audio.

## Requirements

Python 3.12. The system Python on macOS is 3.9 and will not work — use `uv`.

## Setup

    cp ../.env.example ../.env    # then set GROQ_API_KEY
    uv run --python 3.12 --extra dev pytest

## Usage

    uv run --python 3.12 newsninja --topic "artificial intelligence" --topic "climate"

## Design notes

**Structured output.** Extraction uses Groq's strict `json_schema` mode, which is
supported only on `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. Pydantic's schema
output needs a transform first — see `analysis/schema.py`.

**Grounding.** Every `Claim` carries a `quote` that must appear verbatim in its source
article, which makes hallucination a string containment check rather than a judgement.

**Rate limits.** The free tier is tokens-per-minute limited. Work is spread across models
(limits are per model) and articles are batched one call per topic rather than one per
article. See `analysis/limiter.py`.

**Audio.** gTTS by default across twelve languages. Orpheus is an opt-in upgrade covering
English and Saudi Arabic only, and requires accepting model terms at console.groq.com.
Failures fall back to gTTS.

**Reddit.** Requires a free script app at reddit.com/prefs/apps. Without credentials the
source reports itself unavailable; the public JSON endpoint returns 403.
```

- [ ] **Step 8: Commit**

```bash
git add api/newsninja/cli.py api/README.md .github/workflows/ci.yml api/tests/
git commit -m "Add CLI, CI workflow, and package README

CI runs ruff, mypy, and pytest offline with a dummy key. The CLI makes the
pipeline runnable without a web server, which is what lets the eval harness
exercise it directly in Plan 2."
```

---

## Self-Review

**Spec coverage.** Section 3 architecture → Tasks 1–12. Section 4 tiered routing and
limiter → Tasks 4, 5, 9, 10. Section 5 AI core, schemas, retries, prompt versioning →
Tasks 2, 3, 5, 9. Multi-language translation → Task 10. Orpheus routing → Task 11. Reddit
via official API → Task 7. Section 8 error handling → Tasks 2, 6, 7, 12. Section 9 testing
and CI → every task, plus Task 13.

**Deferred to later plans, by design:** the eval harness (§6) is Plan 2; the FastAPI
service and deployment (§3, §11) are Plan 3; the frontend (§7) is Plan 4; the root README
rewrite and LICENSE (§10) belong with Plan 3, when there is a deployed URL to document.

**Known gap accepted:** `newsninja/api.py` appears in the spec's tree but is deliberately
absent here — it is Plan 3's deliverable.

**Type consistency checked.** `Source.fetch(topic, limit)` matches both implementations
and the pipeline's call site. `client.structured(model=, system=, user=, schema_model=,
max_retries=)` is keyword-only and identical across `extract.py` and both test stubs.
`client.text(model=, system=, user=)` matches `synthesize.py` and its stub.
`Cache.make_key(**parts)` and `Cache.get/set` match `extract.py`. `synthesize_speech`'s
signature matches `_default_tts`'s call.

**One trap defused:** the obvious way to write the CLI's `--no-audio` handling is
`tts=... if args.no_audio else None`, which would override `run_pipeline`'s default with
`None` and crash on call. Task 13 builds a kwargs dict and omits the key instead, with a
comment saying why.
