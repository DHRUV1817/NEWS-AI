import pytest
from pydantic import BaseModel

from newsninja.analysis.client import GroqClient, GroqTransport, parse_duration
from newsninja.analysis.limiter import TokenBudgetLimiter
from newsninja.errors import ExtractionFailure, RateLimitError


class Tiny(BaseModel):
    name: str
    score: float


class StubTransport:
    """Stands in for the Groq SDK. Returns queued payloads in order."""

    def __init__(
        self,
        payloads: list[str],
        usage: dict[str, int] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payloads = list(payloads)
        self.requests: list[dict] = []
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 5}
        self.headers = headers or {}

    def complete(self, **kwargs) -> tuple[str, dict[str, int], dict[str, str]]:
        self.requests.append(kwargs)
        if not self.payloads:
            raise AssertionError("transport called more times than expected")
        return self.payloads.pop(0), dict(self.usage), dict(self.headers)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


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


class _FakeResponse:
    """Stands in for the ``response`` attribute the Groq SDK's RateLimitError carries."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class _FakeGroqRateLimitError(Exception):
    """Stands in for ``groq.RateLimitError`` without importing the real SDK.

    The real exception carries ``status_code`` and a ``response`` with headers;
    that shape is all ``GroqTransport.complete`` should rely on.
    """

    def __init__(self, message: str, status_code: int, headers: dict[str, str]) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = _FakeResponse(headers)


class _RateLimitedSDKClient:
    """A fake ``groq.Groq``-shaped client whose raw-response ``create`` always 429s."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.chat = self
        self.completions = self
        self.with_raw_response = self

    def create(self, **kwargs):
        raise self._exc


def test_groq_transport_reraises_rate_limit_as_typed_error():
    fake_exc = _FakeGroqRateLimitError(
        "rate limited", status_code=429, headers={"retry-after": "12.5"}
    )
    transport = GroqTransport.__new__(GroqTransport)
    transport._client = _RateLimitedSDKClient(fake_exc)

    with pytest.raises(RateLimitError) as exc_info:
        transport.complete(model="openai/gpt-oss-20b", messages=[])

    assert exc_info.value.retry_after == 12.5


def test_groq_transport_defaults_retry_after_when_header_missing():
    fake_exc = _FakeGroqRateLimitError("rate limited", status_code=429, headers={})
    transport = GroqTransport.__new__(GroqTransport)
    transport._client = _RateLimitedSDKClient(fake_exc)

    with pytest.raises(RateLimitError) as exc_info:
        transport.complete(model="openai/gpt-oss-20b", messages=[])

    assert exc_info.value.retry_after == 60.0


def test_groq_transport_does_not_mistake_other_errors_for_rate_limits():
    other_exc = ValueError("something unrelated")
    transport = GroqTransport.__new__(GroqTransport)
    transport._client = _RateLimitedSDKClient(other_exc)

    with pytest.raises(ValueError, match="something unrelated"):
        transport.complete(model="openai/gpt-oss-20b", messages=[])


class _FakeMessage:
    content = '{"name": "a", "score": 0.2}'


class _FakeChoice:
    message = _FakeMessage()


class _FakeUsage:
    prompt_tokens = 700
    completion_tokens = 1400


class _FakeParsed:
    """The shape of a parsed Groq chat completion."""

    def __init__(self) -> None:
        self.choices = [_FakeChoice()]
        self.usage = _FakeUsage()


class _RawResponse:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers

    def parse(self):
        return _FakeParsed()


class _RawSDKClient:
    """A fake ``groq.Groq`` whose ``with_raw_response.create`` returns headers."""

    def __init__(self, headers: dict[str, str]) -> None:
        self._headers = headers
        self.chat = self
        self.completions = self
        self.with_raw_response = self

    def create(self, **kwargs):
        return _RawResponse(self._headers)


def test_groq_transport_returns_the_response_headers():
    """Regression: headers were discarded, so `observe()` had no production caller."""
    transport = GroqTransport.__new__(GroqTransport)
    transport._client = _RawSDKClient(
        {"X-RateLimit-Remaining-Tokens": "120", "x-ratelimit-reset-tokens": "7.66s"}
    )

    content, usage, headers = transport.complete(model="openai/gpt-oss-20b", messages=[])

    assert content.startswith("{")
    assert usage == {"prompt_tokens": 700, "completion_tokens": 1400}
    assert headers["x-ratelimit-remaining-tokens"] == "120", "header names are lowercased"
    assert headers["x-ratelimit-reset-tokens"] == "7.66s"


# --- budget accounting ---


def _budgeted_client(transport, tpm=8_000):
    clock = FakeClock()
    limiter = TokenBudgetLimiter(tpm=tpm, clock=clock.time, sleeper=clock.sleep)
    client = GroqClient(
        api_key="test",
        limiters={"openai/gpt-oss-20b": limiter},
        transport=transport,
    )
    return client, limiter, clock


def test_repeated_calls_respect_the_tpm_ceiling():
    """Regression: reserving only `len(prompt)//4` under-counted by ~3x.

    Four calls at 700 prompt + 1,400 completion tokens are 8,400 real tokens,
    past the 8,000 ceiling — the limiter must wait rather than sail through.
    """
    transport = StubTransport(
        ['{"name": "a", "score": 0.1}'] * 10,
        usage={"prompt_tokens": 700, "completion_tokens": 1400},
    )
    client, _limiter, clock = _budgeted_client(transport)

    for _ in range(4):
        client.structured(
            model="openai/gpt-oss-20b",
            system="s" * 1400,
            user="u" * 1400,
            schema_model=Tiny,
        )

    assert clock.slept, "the limiter must wait before breaching the per-minute cap"


def test_actual_usage_replaces_the_estimate_in_the_window():
    transport = StubTransport(
        ['{"name": "a", "score": 0.1}'],
        usage={"prompt_tokens": 3000, "completion_tokens": 2000},
    )
    client, limiter, _clock = _budgeted_client(transport)

    client.structured(model="openai/gpt-oss-20b", system="s", user="u", schema_model=Tiny)

    assert limiter.used_tokens() == 5000, (
        "the window must hold the real 5,000 tokens, not the pre-call estimate"
    )


def test_provider_headers_reach_the_limiter():
    """Regression: `observe()` had zero production callers."""
    transport = StubTransport(
        ['{"name": "a", "score": 0.1}', '{"name": "b", "score": 0.2}'],
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        headers={
            "x-ratelimit-remaining-tokens": "40",  # far below 5% of 8,000
            "x-ratelimit-reset-tokens": "1m5s",
        },
    )
    client, _limiter, clock = _budgeted_client(transport)

    client.structured(model="openai/gpt-oss-20b", system="s", user="u", schema_model=Tiny)
    assert clock.slept == [], "the first call has nothing to wait for"

    client.structured(model="openai/gpt-oss-20b", system="s", user="u", schema_model=Tiny)
    assert clock.slept == [65.0], "the provider's stated reset must be honoured"


def test_a_model_with_no_configured_budget_is_an_error_not_a_silent_pass():
    client = GroqClient(api_key="test", transport=StubTransport([]))
    client._limiters = {}
    with pytest.raises(ValueError, match="MODEL_TPM"):
        client.text(model="openai/gpt-oss-20b", system="s", user="u")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("7.66s", 7.66), ("2m59.56s", 179.56), ("500ms", 0.5), ("12.5", 12.5)],
)
def test_duration_headers_are_parsed(value, expected):
    assert parse_duration(value) == pytest.approx(expected)


def test_unparseable_duration_falls_back_rather_than_returning_zero():
    assert parse_duration(None) == 60.0
    assert parse_duration("soon") == 60.0


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
