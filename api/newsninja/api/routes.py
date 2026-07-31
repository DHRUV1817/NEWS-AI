"""The endpoints.

Split along seams the package already has, so no single request runs long
enough for a platform proxy to sever it. See
docs/superpowers/specs/2026-07-31-fastapi-service-design.md for the token
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
