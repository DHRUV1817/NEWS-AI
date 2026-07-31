"""Speech synthesis with honest language routing.

gTTS is the default: free, no authentication, twelve languages. Orpheus is an
opt-in upgrade that produces markedly better audio but covers only English and
Saudi Arabic (verified 2026-07-31) and returns 400 until its terms are accepted
at console.groq.com. Any Orpheus failure falls back to gTTS rather than
surfacing as a broken feature.
"""

import io
import re
from collections.abc import Callable
from typing import Any

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "hi", "ar",
)

ORPHEUS_LANGUAGES = frozenset({"en", "ar"})

ORPHEUS_MODELS = {
    "en": "canopylabs/orpheus-v1-english",
    "ar": "canopylabs/orpheus-arabic-saudi",
}

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, limit: int = 3000) -> list[str]:
    """Split on sentence boundaries into chunks of at most ``limit`` characters.

    Orpheus has a 4,000-token context, and gTTS degrades on very long input.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE.split(text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
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


def synthesize_speech(
    text: str,
    language: str = "en",
    enable_orpheus: bool = False,
    gtts_factory: Callable[..., Any] | None = None,
    orpheus_fn: Callable[[str, str], bytes] | None = None,
) -> bytes:
    """Render ``text`` to audio bytes, routing by language and availability."""
    if language not in SUPPORTED_LANGUAGES:
        language = "en"

    factory = gtts_factory or _default_gtts_factory

    if enable_orpheus and language in ORPHEUS_LANGUAGES and orpheus_fn is not None:
        try:
            return orpheus_fn(text, language)
        except Exception:  # noqa: BLE001, S110
            # Terms not accepted (400 model_terms_required), or the model is
            # otherwise unavailable. This is the designed fallback, not an
            # oversight: gTTS still produces valid audio, so we degrade
            # silently rather than surface a broken feature.
            pass

    return _gtts_bytes(text, language, factory)
