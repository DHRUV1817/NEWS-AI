"""Orchestration: topics in, briefing and audio out.

Source failures are collected per source rather than aborting the run, so one
dead source degrades the briefing instead of destroying it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from newsninja.analysis.client import StructuredClient, TextClient
from newsninja.analysis.extract import extract_topic
from newsninja.analysis.synthesize import build_briefing
from newsninja.audio.tts import synthesize_speech
from newsninja.cache import Cache
from newsninja.errors import SourceError
from newsninja.models import Article, Briefing

if TYPE_CHECKING:
    from newsninja.sources.base import Source

MAX_TOPICS = 5


class PipelineClient(StructuredClient, TextClient, Protocol):
    """Union seam: ``run_pipeline`` needs both ``structured()`` (for topic
    extraction) and ``text()`` (for briefing synthesis) from one client.

    Satisfied structurally by ``GroqClient`` and by test stubs without either
    needing to subclass this.
    """


@dataclass
class PipelineResult:
    briefing: Briefing
    audio: bytes
    source_errors: dict[str, str] = field(default_factory=dict)


def _default_tts(text: str, language: str, enable_orpheus: bool) -> bytes:
    return synthesize_speech(text, language=language, enable_orpheus=enable_orpheus)


def run_pipeline(
    topics: list[str],
    sources: list[Source],
    client: PipelineClient,
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
