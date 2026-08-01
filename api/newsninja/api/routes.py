"""The endpoints.

Split along seams the package already has, so no single request runs long
enough for a platform proxy to sever it. See
docs/design/specs/2026-07-31-fastapi-service-design.md for the token
arithmetic that forces this shape.

Every handler is a plain ``def`` rather than ``async def``, so FastAPI runs it
in a threadpool. The token limiter blocks with ``time.sleep``; on the event
loop that would stall every other request in the process.
"""

from collections.abc import Callable
from importlib.metadata import version
from typing import Annotated

from fastapi import APIRouter, Depends, Response

from newsninja.analysis.client import GroqClient
from newsninja.analysis.synthesize import build_briefing
from newsninja.api.deps import get_cache, get_client, get_sources, get_tts
from newsninja.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AudioRequest,
    BriefRequest,
    BriefResponse,
    HealthResponse,
)
from newsninja.audio.tts import SpokenAudio
from newsninja.cache import Cache
from newsninja.config import Settings, get_settings
from newsninja.pipeline import analyze_topic
from newsninja.sources.base import Source

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only.

    Deliberately does not call Groq. A health check that spends tokens against
    an 8,000 TPM budget is a health check that causes outages.
    """
    return HealthResponse(status="ok", version=version("newsninja"))


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


@router.post("/audio")
def audio(
    payload: AudioRequest,
    tts: Annotated[Callable[[str, str, bool], SpokenAudio], Depends(get_tts)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Render a supplied script to speech.

    Returns raw bytes rather than JSON, with the media type the speech seam
    reports for the bytes it produced. Choosing it from ENABLE_ORPHEUS instead
    would be a guess: every Orpheus failure falls back to gTTS, which returns
    mp3, and on an account that has not accepted the model terms that fallback
    is the live path — so the header would say audio/wav over mp3 bytes.
    """
    spoken = tts(payload.script, payload.language, settings.enable_orpheus)
    return Response(content=spoken.data, media_type=spoken.media_type)
