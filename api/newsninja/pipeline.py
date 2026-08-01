"""Orchestration: topics in, briefing and audio out.

Source failures are collected per source rather than aborting the run, so one
dead source degrades the briefing instead of destroying it. Sources that report
themselves unavailable are recorded separately: a missing credential is not a
failure, but the user is still entitled to know the briefing was built from
fewer sources than they asked for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from newsninja.analysis.client import StructuredClient, TextClient
from newsninja.analysis.extract import extract_topic
from newsninja.analysis.synthesize import build_briefing
from newsninja.audio.tts import (
    SpokenAudio,
    make_orpheus_fn,
    render_speech,
    synthesize_speech,
)
from newsninja.cache import Cache
from newsninja.errors import SourceError
from newsninja.models import Article, ArticleAnalysis, Briefing

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
    #: Sources that raised, by name. One entry per failure, so a source failing
    #: on several topics keeps every message rather than only the last.
    source_errors: dict[str, list[str]] = field(default_factory=dict)
    #: Sources skipped because they reported themselves unavailable — missing
    #: Reddit credentials, typically. Not an error, but the user is entitled to
    #: know the briefing was built from fewer sources than they expected.
    skipped_sources: list[str] = field(default_factory=list)


def _orpheus_seam(
    api_key: str | None, enable_orpheus: bool
) -> Callable[[str, str], bytes] | None:
    """Wire Orpheus only when it can actually run.

    Orpheus needs a Groq key of its own — it is a separate REST endpoint, not
    part of the chat client — so without one the branch is not wired at all and
    gTTS handles every language.
    """
    return make_orpheus_fn(api_key) if enable_orpheus and api_key else None


def default_tts(api_key: str | None = None) -> Callable[[str, str, bool], bytes]:
    """Build the speech seam ``run_pipeline`` uses when none is injected.

    Bytes only: the CLI writes them to the path the user named. A caller that
    must describe the bytes wants ``default_speech``.
    """

    def _tts(text: str, language: str, enable_orpheus: bool) -> bytes:
        return synthesize_speech(
            text,
            language=language,
            enable_orpheus=enable_orpheus,
            orpheus_fn=_orpheus_seam(api_key, enable_orpheus),
        )

    return _tts


def default_speech(api_key: str | None = None) -> Callable[[str, str, bool], SpokenAudio]:
    """The same seam, reporting the format it produced alongside the bytes.

    The HTTP layer uses this one. ``enable_orpheus=True`` does not imply wav —
    every Orpheus failure falls back to gTTS — so a Content-Type chosen from
    configuration would label mp3 bytes ``audio/wav``.
    """

    def _speech(text: str, language: str, enable_orpheus: bool) -> SpokenAudio:
        return render_speech(
            text,
            language=language,
            enable_orpheus=enable_orpheus,
            orpheus_fn=_orpheus_seam(api_key, enable_orpheus),
        )

    return _speech


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
    #: How many articles the analysis was built from. Zero means no model call
    #: happened at all: ``extract_topic`` short-circuits an empty list to a
    #: placeholder carrying ``stance="neutral"`` and ``confidence=0.0``, and
    #: without this count a caller cannot tell that verdict from a measured one.
    article_count: int = 0


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
        article_count=len(articles),
    )


def run_pipeline(
    topics: list[str],
    sources: list[Source],
    client: PipelineClient,
    cache: Cache | None = None,
    language: str = "en",
    enable_orpheus: bool = False,
    limit: int = 8,
    tts: Callable[[str, str, bool], bytes] | None = None,
    api_key: str | None = None,
) -> PipelineResult:
    """Fetch, extract, synthesise, and render audio for ``topics``."""
    if not topics:
        raise ValueError("at least one topic is required")
    if len(topics) > MAX_TOPICS:
        raise ValueError(f"at most {MAX_TOPICS} topics per run, got {len(topics)}")

    source_errors: dict[str, list[str]] = {}
    skipped_sources: list[str] = []
    analyses = []

    for topic in topics:
        outcome = analyze_topic(topic, sources, client, cache=cache, limit=limit)
        analyses.append(outcome.analysis)
        for name, messages in outcome.source_errors.items():
            source_errors.setdefault(name, []).extend(messages)
        for name in outcome.skipped_sources:
            if name not in skipped_sources:
                skipped_sources.append(name)

    briefing = build_briefing(client, analyses, language=language)
    render = tts if tts is not None else default_tts(api_key)
    audio = render(briefing.script, language, enable_orpheus)

    return PipelineResult(
        briefing=briefing,
        audio=audio,
        source_errors=source_errors,
        skipped_sources=skipped_sources,
    )
