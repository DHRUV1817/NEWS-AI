"""The only module in this package that talks to Groq.

Responsibilities: strict-schema calls, validation-feedback retries, token
budgeting, and usage accounting. Keeping the SDK behind this seam means a
provider swap or a test stub touches exactly one file.
"""

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from newsninja.analysis.limiter import TokenBudgetLimiter
from newsninja.analysis.schema import strict_schema
from newsninja.errors import ExtractionFailure, RateLimitError

T = TypeVar("T", bound=BaseModel)

# Verified 2026-07-31: only these accept response_format=json_schema.
SCHEMA_CAPABLE_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b"})

# Verified free-tier tokens-per-minute ceilings.
MODEL_TPM = {
    "openai/gpt-oss-20b": 8_000,
    "openai/gpt-oss-120b": 8_000,
    "llama-3.3-70b-versatile": 12_000,
}

DEFAULT_RETRY_AFTER = 60.0


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0


class Transport(Protocol):
    """Minimal seam over the Groq SDK so tests need no network."""

    def complete(self, **kwargs: Any) -> tuple[str, dict[str, int]]:
        """Return (content, usage_dict)."""


class StructuredClient(Protocol):
    """Structural seam for callers that only need ``structured()``.

    Satisfied by ``GroqClient`` and by test stubs (e.g. ``RecordingClient``)
    without either needing to subclass this — it's a ``Protocol``, matched
    structurally, not a nominal base class.
    """

    def structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema_model: type[T],
        max_retries: int = 2,
    ) -> T: ...


class TextClient(Protocol):
    """Structural seam for callers that only need free-form ``text()``.

    Satisfied by ``GroqClient`` and by test stubs without either needing to
    subclass this — it's a ``Protocol``, matched structurally, not a nominal
    base class.
    """

    def text(self, *, model: str, system: str, user: str) -> str: ...


def _rate_limit_retry_after(exc: Exception) -> float:
    """Extract ``retry-after`` (seconds) from a Groq SDK rate-limit exception.

    Falls back to ``DEFAULT_RETRY_AFTER`` when the header is absent or the
    value cannot be parsed as a float.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("retry-after") if headers is not None else None
    if value is None:
        return DEFAULT_RETRY_AFTER
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULT_RETRY_AFTER


class GroqTransport:
    def __init__(self, api_key: str) -> None:
        from groq import Groq

        self._client = Groq(api_key=api_key)

    def complete(self, **kwargs: Any) -> tuple[str, dict[str, int]]:
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # The Groq SDK raises ``groq.RateLimitError`` (status_code=429,
            # a ``response`` carrying headers) on a 429. Re-raise that as our
            # own typed error so callers never need to import the SDK to
            # catch it; anything else propagates unchanged.
            if getattr(exc, "status_code", None) == 429:
                raise RateLimitError(str(exc), _rate_limit_retry_after(exc)) from exc
            raise
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
