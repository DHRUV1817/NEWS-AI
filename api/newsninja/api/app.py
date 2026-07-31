"""Application assembly.

``create_app`` exists because middleware is configured from settings at
construction time. Adding CORS at module scope would read the environment once,
at import, and no later configuration could change it — which makes the
allowlist both untestable and unchangeable after the first import.

There is deliberately no module-level ``app`` instance. Constructing one at
import time would call ``get_settings()`` on any import of anything under
``newsninja.api`` — including a plain test collection pass — and that reads
credentials from the environment or ``.env``. On a machine with no configured
key that turns "import this package" into a crash, and on a machine that does
have one it means merely importing the package pulls a real key into memory.
The server is started as a factory instead:

    uvicorn newsninja.api:create_app --factory
"""

import math
from collections.abc import Awaitable, Callable
from importlib.metadata import version

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from newsninja.api.ratelimit import RateLimiter
from newsninja.api.routes import router
from newsninja.config import Settings, get_settings
from newsninja.errors import ExtractionFailure, RateLimitError, SourceError


def _error(
    status: int,
    kind: str,
    message: str,
    retry_after: float | None = None,
    detail: object | None = None,
) -> JSONResponse:
    """One envelope for every failure, so a caller parses one shape."""
    body: dict[str, object] = {"type": kind, "message": message}
    # Both annotations are required: mypy --strict rejects a bare `{}`.
    headers: dict[str, str] = {}
    if retry_after is not None:
        body["retry_after"] = retry_after
        # delta-seconds is an integer, and rounding up matters: truncating 0.4
        # to "0" tells the caller to retry straight back into a full budget.
        headers["Retry-After"] = str(math.ceil(retry_after))
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(status_code=status, content={"error": body}, headers=headers)


async def _handle_rate_limit(request: Request, exc: Exception) -> JSONResponse:
    # Typed as Exception because that is the signature Starlette's registry
    # declares. Narrowing in the parameter list would need a mypy suppression
    # comment, and this project keeps that count at zero.
    retry_after = exc.retry_after if isinstance(exc, RateLimitError) else None
    return _error(429, "rate_limit", str(exc), retry_after=retry_after)


async def _handle_source_error(request: Request, exc: Exception) -> JSONResponse:
    return _error(502, "source_error", str(exc))


async def _handle_extraction_failure(request: Request, exc: Exception) -> JSONResponse:
    return _error(502, "extraction_failure", str(exc))


async def _handle_value_error(request: Request, exc: Exception) -> JSONResponse:
    return _error(422, "invalid_request", str(exc))


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    # Typed as Exception to match the handler style above; narrowed here.
    # RequestValidationError does not subclass ValueError, and FastAPI
    # pre-registers its own handler for it, so without this override a bad
    # request body would answer in FastAPI's shape rather than this service's
    # one envelope.
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    # exc.errors() can carry non-JSON-serialisable values (a raised ValueError
    # in `ctx`, for a field validator's error) — jsonable_encoder coerces
    # those to something JSONResponse can render instead of the handler
    # itself 500ing on encode.
    return _error(
        422, "invalid_request", "request validation failed", detail=jsonable_encoder(errors)
    )


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
    # Registered explicitly to override FastAPI's own pre-registered handler
    # for this exact exception type, so a bad request body still answers in
    # this service's one envelope rather than FastAPI's `{"detail": [...]}`.
    application.add_exception_handler(RequestValidationError, _handle_validation_error)

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
