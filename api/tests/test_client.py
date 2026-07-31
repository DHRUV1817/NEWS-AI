import pytest
from pydantic import BaseModel

from newsninja.analysis.client import GroqClient, GroqTransport
from newsninja.errors import ExtractionFailure, RateLimitError


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
    """A fake ``groq.Groq``-shaped client whose ``chat.completions.create`` always 429s."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.chat = self
        self.completions = self

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
