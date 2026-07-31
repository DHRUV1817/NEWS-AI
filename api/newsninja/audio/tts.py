"""Speech synthesis with honest language routing.

gTTS is the default: free, no authentication, twelve languages. Orpheus is an
opt-in upgrade that produces markedly better audio but covers only English and
Saudi Arabic (verified 2026-07-31) and returns 400 ``model_terms_required``
until its terms are accepted at console.groq.com. Any Orpheus failure falls
back to gTTS rather than surfacing as a broken feature — on an account that has
not accepted the terms, that fallback is the live path.
"""

import io
import re
import wave
from collections.abc import Callable
from typing import Any, NamedTuple

#: gTTS returns mp3, Orpheus returns wav. Which one ran is not decidable from
#: configuration — see ``SpokenAudio``.
MP3_MEDIA_TYPE = "audio/mpeg"
WAV_MEDIA_TYPE = "audio/wav"


class SpokenAudio(NamedTuple):
    """Rendered speech together with the media type of the bytes produced.

    The two travel as one value because the engine that ran is not the engine
    that was asked for: every Orpheus failure falls back to gTTS, and on an
    account that has not accepted the model terms that fallback is the live
    path. A caller choosing a header from its own configuration would then label
    mp3 bytes ``audio/wav``.
    """

    data: bytes
    media_type: str


SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "hi", "ar",
)

ORPHEUS_LANGUAGES = frozenset({"en", "ar"})

ORPHEUS_MODELS = {
    "en": "canopylabs/orpheus-v1-english",
    "ar": "canopylabs/orpheus-arabic-saudi",
}

ORPHEUS_ENDPOINT = "https://api.groq.com/openai/v1/audio/speech"
ORPHEUS_VOICE = "tara"
ORPHEUS_TIMEOUT = 60

# Sentence terminators. The Latin set requires following whitespace so that
# "3.5" and "e.g." are not treated as boundaries. CJK and Arabic terminators
# are unambiguous and are routinely written with no following space, so they
# split on a zero-width boundary — without this, a page of Chinese is a single
# unsplittable segment.
_SENTENCE_END = re.compile(r"[.!?]\s+|[。！？؟۔]\s*")

# ``post(url, headers, payload) -> audio bytes``. Injecting this keeps every
# test off the network; the default implementation is the only place speech
# synthesis touches httpx.
OrpheusPost = Callable[[str, dict[str, str], dict[str, Any]], bytes]


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, each keeping its own trailing whitespace.

    Concatenating the result reproduces ``text`` exactly, which is what lets
    ``chunk_text`` promise that no content is lost.
    """
    segments: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if end == start:  # zero-width match at a position already consumed
            continue
        segments.append(text[start:end])
        start = end
    if start < len(text):
        segments.append(text[start:])
    return segments


def _hard_split(segment: str, limit: int) -> list[str]:
    """Cut ``segment`` into pieces of at most ``limit`` characters.

    Prefers the last space inside the window so words survive, and falls back to
    a blunt cut for scripts that do not use spaces. Concatenating the pieces
    reproduces ``segment`` exactly.
    """
    if len(segment) <= limit:
        return [segment]

    pieces: list[str] = []
    rest = segment
    while len(rest) > limit:
        cut = rest.rfind(" ", 0, limit)
        cut = limit if cut <= 0 else cut + 1  # keep the space on the left piece
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)
    return pieces


def chunk_text(text: str, limit: int = 3000) -> list[str]:
    """Split on sentence boundaries into chunks of at most ``limit`` characters.

    Orpheus has a 4,000-token context and gTTS degrades on very long input, so
    the length bound is a guarantee rather than a preference: a segment with no
    usable boundary — one long unpunctuated paragraph, or a script this splitter
    does not know — is cut bluntly instead of being returned oversized.
    ``"".join(chunk_text(t, n))`` equals ``t.strip()``: nothing is dropped,
    reordered, or duplicated.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")

    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for segment in _split_sentences(text):
        for piece in _hard_split(segment, limit):
            if len(current) + len(piece) <= limit:
                current += piece
            else:
                if current:
                    chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def _gtts_bytes(text: str, language: str, factory: Callable[..., Any]) -> bytes:
    buffer = io.BytesIO()
    for chunk in chunk_text(text):
        factory(text=chunk, lang=language, slow=False).write_to_fp(buffer)
    return buffer.getvalue()


def _default_gtts_factory(**kwargs: Any) -> Any:
    from gtts import gTTS

    return gTTS(**kwargs)


def _default_orpheus_post(
    url: str, headers: dict[str, str], payload: dict[str, Any]
) -> bytes:
    import httpx

    response = httpx.post(url, headers=headers, json=payload, timeout=ORPHEUS_TIMEOUT)
    response.raise_for_status()
    return response.content


def _concat_wav(parts: list[bytes]) -> bytes:
    """Join WAV payloads into one stream.

    Raw byte concatenation would bury a RIFF header mid-file and most players
    would render only the first chunk, so the frames are re-muxed under a single
    header instead.
    """
    if not parts:
        return b""
    if len(parts) == 1:
        return parts[0]

    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        for index, part in enumerate(parts):
            with wave.open(io.BytesIO(part), "rb") as reader:
                if index == 0:
                    writer.setparams(reader.getparams())
                writer.writeframes(reader.readframes(reader.getnframes()))
    return out.getvalue()


def orpheus_speech(
    text: str,
    language: str,
    api_key: str,
    post: OrpheusPost = _default_orpheus_post,
) -> bytes:
    """Synthesise ``text`` with Orpheus on Groq and return WAV bytes.

    Long input is chunked, one request per chunk, and the responses are re-muxed
    into a single WAV. Every failure propagates — including the HTTP 400
    ``model_terms_required`` an account receives until it accepts the model
    terms — so that ``synthesize_speech`` can fall back to gTTS.
    """
    model = ORPHEUS_MODELS.get(language)
    if model is None:
        raise ValueError(
            f"Orpheus does not cover {language!r}; supported: {sorted(ORPHEUS_MODELS)}"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    parts = [
        post(
            ORPHEUS_ENDPOINT,
            headers,
            {
                "model": model,
                "input": chunk,
                "voice": ORPHEUS_VOICE,
                "response_format": "wav",
            },
        )
        for chunk in chunk_text(text)
    ]
    return _concat_wav(parts)


def make_orpheus_fn(
    api_key: str, post: OrpheusPost = _default_orpheus_post
) -> Callable[[str, str], bytes]:
    """Bind ``api_key`` into the ``orpheus_fn`` seam ``synthesize_speech`` takes."""

    def _orpheus_fn(text: str, language: str) -> bytes:
        return orpheus_speech(text, language, api_key=api_key, post=post)

    return _orpheus_fn


def render_speech(
    text: str,
    language: str = "en",
    enable_orpheus: bool = False,
    gtts_factory: Callable[..., Any] | None = None,
    orpheus_fn: Callable[[str, str], bytes] | None = None,
) -> SpokenAudio:
    """Render ``text`` to audio, reporting which format actually came out.

    Routing by language and availability is one thing; describing the result is
    another, and only this function knows both. ``enable_orpheus=True`` does not
    mean wav: the fallback below is silent by design and, on an account without
    accepted terms, is the path that always runs.
    """
    if language not in SUPPORTED_LANGUAGES:
        language = "en"

    factory = gtts_factory or _default_gtts_factory

    if enable_orpheus and language in ORPHEUS_LANGUAGES and orpheus_fn is not None:
        try:
            return SpokenAudio(orpheus_fn(text, language), WAV_MEDIA_TYPE)
        except Exception:  # noqa: BLE001, S110
            # Terms not accepted (400 model_terms_required), or the model is
            # otherwise unavailable. This is the designed fallback, not an
            # oversight: gTTS still produces valid audio, so we degrade
            # silently rather than surface a broken feature.
            pass

    return SpokenAudio(_gtts_bytes(text, language, factory), MP3_MEDIA_TYPE)


def synthesize_speech(
    text: str,
    language: str = "en",
    enable_orpheus: bool = False,
    gtts_factory: Callable[..., Any] | None = None,
    orpheus_fn: Callable[[str, str], bytes] | None = None,
) -> bytes:
    """Render ``text`` to audio bytes, routing by language and availability.

    Bytes only, for callers that write them to a file the user named. A caller
    that has to *describe* the bytes — the HTTP layer setting a Content-Type —
    wants ``render_speech`` instead.
    """
    return render_speech(
        text, language, enable_orpheus, gtts_factory, orpheus_fn
    ).data
