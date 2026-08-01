"""The only module in this package that talks to Groq.

Responsibilities: strict-schema calls, validation-feedback retries, token
budgeting, and usage accounting. Keeping the SDK behind this seam means a
provider swap or a test stub touches exactly one file.
"""

import re
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from newsninja.analysis.limiter import TokenBudgetLimiter
from newsninja.analysis.schema import strict_schema
from newsninja.errors import ExtractionFailure, RateLimitError, SchemaRejection

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

# Measured on structured extraction against gpt-oss-20b: a batched topic call
# returns roughly 1,200-1,800 completion tokens. Reserving the prompt alone made
# ten extractions read as ~6,000 tokens locally while really spending ~21,000 —
# straight through the 8,000 TPM ceiling without the limiter ever waiting.
EXPECTED_COMPLETION_TOKENS = 1_500

# Groq's rate-limit headers. Values are durations like "7.66s" or "2m59.56s".
REMAINING_TOKENS_HEADER = "x-ratelimit-remaining-tokens"
RESET_TOKENS_HEADER = "x-ratelimit-reset-tokens"

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)?")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0


class Transport(Protocol):
    """Minimal seam over the Groq SDK so tests need no network.

    Response headers are part of the return value because the provider's
    ``x-ratelimit-*`` headers are the only ground truth about the budget; a
    transport that discarded them would leave the limiter guessing.
    """

    def complete(self, **kwargs: Any) -> tuple[str, dict[str, int], dict[str, str]]:
        """Return (content, usage_dict, response_headers)."""


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


def parse_duration(value: str | None, default: float = DEFAULT_RETRY_AFTER) -> float:
    """Parse a Groq duration header into seconds.

    Handles the shapes the API actually emits — ``"7.66s"``, ``"2m59.56s"``,
    ``"500ms"`` and a bare number — and falls back to ``default`` for anything
    it cannot read rather than pretending the wait is zero.
    """
    if value is None:
        return default
    parts = _DURATION_PART.findall(value)
    if not parts:
        return default
    return sum(float(amount) * _UNIT_SECONDS[unit or "s"] for amount, unit in parts)


def _rate_limit_retry_after(exc: Exception) -> float:
    """Extract ``retry-after`` (seconds) from a Groq SDK rate-limit exception.

    Falls back to ``DEFAULT_RETRY_AFTER`` when the header is absent or the
    value cannot be parsed.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("retry-after") if headers is not None else None
    return parse_duration(value)


def _schema_rejection_details(exc: Exception) -> tuple[str, str] | None:
    """Pull ``(message, failed_generation)`` out of a provider schema-rejection.

    Groq answers a strict-schema mismatch with an HTTP 400 whose body carries
    ``code: "json_validate_failed"`` and the offending generation. The SDK
    exposes that parsed body on ``exc.body``, but its shape is not part of any
    contract, so every step here is defensive: returns ``None`` — meaning "not
    a schema rejection" — for anything that does not match exactly, so a 400
    for an unrelated reason (a malformed schema, a bad model name) still
    propagates unchanged instead of being silently retried.
    """
    if getattr(exc, "status_code", None) != 400:
        return None
    body: Any = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    error: Any = body.get("error")
    if not isinstance(error, dict):
        return None
    if error.get("code") != "json_validate_failed":
        return None
    message = error.get("message")
    if not isinstance(message, str) or not message:
        message = str(exc)
    failed_generation = error.get("failed_generation")
    if not isinstance(failed_generation, str):
        failed_generation = ""
    return message, failed_generation


class GroqTransport:
    def __init__(self, api_key: str) -> None:
        from groq import Groq

        self._client = Groq(api_key=api_key)

    def complete(self, **kwargs: Any) -> tuple[str, dict[str, int], dict[str, str]]:
        try:
            # ``with_raw_response`` keeps the HTTP headers, which the parsed
            # response object throws away. The rate-limit headers on them are
            # what lets the limiter correct itself.
            raw = self._client.chat.completions.with_raw_response.create(**kwargs)
        except Exception as exc:
            # The Groq SDK raises ``groq.RateLimitError`` (status_code=429,
            # a ``response`` carrying headers) on a 429. Re-raise that as our
            # own typed error so callers never need to import the SDK to
            # catch it; anything else propagates unchanged.
            if getattr(exc, "status_code", None) == 429:
                raise RateLimitError(str(exc), _rate_limit_retry_after(exc)) from exc
            # A strict-schema mismatch is a server-side ``ExtractionFailure``:
            # the model produced JSON, the schema refused it, no content came
            # back. That must reach ``GroqClient.structured``'s retry loop the
            # same way a local ``ValidationError`` does, not escape as a raw
            # SDK exception.
            details = _schema_rejection_details(exc)
            if details is not None:
                message, failed_generation = details
                raise SchemaRejection(message, failed_generation) from exc
            raise
        headers = {str(name).lower(): str(value) for name, value in raw.headers.items()}
        response = raw.parse()
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
            headers,
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

    def _limiter_for(self, model: str) -> TokenBudgetLimiter:
        """Fail loudly for a model with no configured budget.

        Silently skipping the limiter for an unlisted model is how a run walks
        into a 429 with no local warning, so an unknown model is a programming
        error rather than a free pass.
        """
        limiter = self._limiters.get(model)
        if limiter is None:
            raise ValueError(
                f"no token budget configured for model {model!r}; "
                f"add it to MODEL_TPM (configured: {sorted(self._limiters)})"
            )
        return limiter

    def _reserve(self, model: str, prompt: str) -> int:
        """Book the prompt *and* the completion the response will cost."""
        estimate = _estimate_tokens(prompt) + EXPECTED_COMPLETION_TOKENS
        return self._limiter_for(model).reserve(estimate, max_wait=self._max_wait)

    def _settle(
        self,
        model: str,
        reservation: int,
        usage: dict[str, int],
        headers: dict[str, str],
    ) -> None:
        """Correct the window with what the call really cost, then with what
        the provider says is left."""
        limiter = self._limiter_for(model)
        limiter.settle(
            reservation,
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        )

        remaining = headers.get(REMAINING_TOKENS_HEADER)
        if remaining is None:
            return
        try:
            remaining_tokens = int(float(remaining))
        except ValueError:
            return
        limiter.observe(
            remaining_tokens, parse_duration(headers.get(RESET_TOKENS_HEADER))
        )

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
            # Booked before the call, not settled if the provider rejects the
            # generation (see the SchemaRejection branch below): the estimate
            # stands for the window on a rejection because the provider still
            # spent those tokens producing the output it then refused.
            reservation = self._reserve(model, system + user)
            try:
                content, usage, headers = self._transport.complete(
                    model=model, messages=messages, response_format=response_format
                )
            except SchemaRejection as exc:
                # The provider's own strict-schema check rejected the
                # generation server-side — no content, no usage, so there is
                # nothing to record or settle. This is the server-side twin of
                # the ValidationError branch below: same retry budget, same
                # correction-message shape, fed from the provider's message
                # and the generation it refused instead of local parsing.
                last_error = str(exc)
                if attempt == max_retries:
                    break
                messages = messages + [
                    {"role": "assistant", "content": exc.failed_generation},
                    {
                        "role": "user",
                        "content": (
                            "That response failed schema validation with the "
                            f"following errors:\n{last_error}\n"
                            "Return corrected JSON matching the schema exactly."
                        ),
                    },
                ]
                continue

            self._record(usage)
            self._settle(model, reservation, usage, headers)

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
        reservation = self._reserve(model, system + user)
        content, usage, headers = self._transport.complete(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        self._record(usage)
        self._settle(model, reservation, usage, headers)
        return content
