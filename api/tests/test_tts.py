import io
import wave
from typing import ClassVar

import pytest

from newsninja.audio.tts import (
    ORPHEUS_ENDPOINT,
    ORPHEUS_LANGUAGES,
    ORPHEUS_MODELS,
    SUPPORTED_LANGUAGES,
    chunk_text,
    make_orpheus_fn,
    orpheus_speech,
    synthesize_speech,
)


class FakeGTTS:
    instances: ClassVar[list["FakeGTTS"]] = []

    def __init__(self, text, lang, slow=False):
        self.text = text
        self.lang = lang
        FakeGTTS.instances.append(self)

    def write_to_fp(self, fp):
        fp.write(b"MP3" + self.text.encode()[:4])


@pytest.fixture(autouse=True)
def _reset():
    FakeGTTS.instances = []


def test_chunk_text_keeps_short_text_whole():
    assert chunk_text("hello there") == ["hello there"]


def test_chunk_text_splits_on_sentence_boundaries():
    text = ". ".join(f"sentence number {i}" for i in range(200)) + "."
    chunks = chunk_text(text, limit=200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_loses_no_words():
    text = ". ".join(f"sentence number {i}" for i in range(50)) + "."
    joined = " ".join(chunk_text(text, limit=120))
    assert joined.replace("  ", " ").split() == text.split()


def test_gtts_is_used_by_default():
    audio = synthesize_speech("hello", language="en", gtts_factory=FakeGTTS)
    assert audio.startswith(b"MP3")
    assert FakeGTTS.instances[0].lang == "en"


def test_unsupported_language_falls_back_to_english():
    synthesize_speech("hello", language="xx", gtts_factory=FakeGTTS)
    assert FakeGTTS.instances[0].lang == "en"


def test_orpheus_used_for_english_when_enabled():
    calls = []

    def orpheus(text, language):
        calls.append(language)
        return b"WAVdata"

    audio = synthesize_speech(
        "hello", language="en", enable_orpheus=True,
        gtts_factory=FakeGTTS, orpheus_fn=orpheus,
    )
    assert audio == b"WAVdata"
    assert calls == ["en"]


def test_orpheus_skipped_for_unsupported_language():
    def orpheus(text, language):
        raise AssertionError("Orpheus must not be called for French")

    synthesize_speech(
        "bonjour", language="fr", enable_orpheus=True,
        gtts_factory=FakeGTTS, orpheus_fn=orpheus,
    )
    assert FakeGTTS.instances[0].lang == "fr"


def test_orpheus_failure_falls_back_to_gtts():
    def orpheus(text, language):
        raise RuntimeError("400 model_terms_required")

    audio = synthesize_speech(
        "hello", language="en", enable_orpheus=True,
        gtts_factory=FakeGTTS, orpheus_fn=orpheus,
    )
    assert audio.startswith(b"MP3")


def test_orpheus_language_set_is_english_and_arabic():
    assert ORPHEUS_LANGUAGES == frozenset({"en", "ar"})


def test_twelve_languages_are_supported():
    assert len(SUPPORTED_LANGUAGES) == 12


# --- chunk_text bounds (regression: both cases returned one oversized chunk) ---


def test_chunk_text_bounds_a_segment_with_no_sentence_boundary():
    """Verified failure: 'word ' * 2000 came back as a single 9,999-char chunk."""
    text = "word " * 2000
    chunks = chunk_text(text, limit=3000)
    assert len(chunks) > 1
    assert all(len(c) <= 3000 for c in chunks)


def test_chunk_text_splits_chinese_on_its_own_terminators():
    """Verified failure: the `(?<=[.!?])\\s+` pattern never matched '。'."""
    text = "新闻很重要。" * 1000  # 6,000 characters, no ASCII punctuation
    chunks = chunk_text(text, limit=1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    assert all(c.endswith("。") for c in chunks), "chunks should break at 。"


def test_chunk_text_splits_arabic_on_its_own_terminators():
    text = ("هذا خبر مهم؟ " * 400).strip()
    chunks = chunk_text(text, limit=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_text_reassembles_exactly():
    """The no-content-loss guarantee, stated as an equality rather than a hope."""
    for text, limit in (
        ("word " * 2000, 3000),
        ("新闻很重要。" * 1000, 1000),
        (". ".join(f"sentence number {i}" for i in range(200)) + ".", 200),
        ("x" * 5000, 700),
    ):
        assert "".join(chunk_text(text, limit=limit)) == text.strip()


def test_chunk_text_rejects_a_nonsensical_limit():
    with pytest.raises(ValueError, match="at least 1"):
        chunk_text("hello", limit=0)


# --- Orpheus (no network: the HTTP callable is injected) ---


def _wav(frames: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(frames)
    return buffer.getvalue()


def _frames(payload: bytes) -> bytes:
    with wave.open(io.BytesIO(payload), "rb") as reader:
        return reader.readframes(reader.getnframes())


class FakePost:
    """Stands in for the HTTP POST. Records every request; makes no network call."""

    def __init__(self, frames: bytes = b"\x01\x02" * 8, error: Exception | None = None):
        self.calls: list[dict] = []
        self._frames = frames
        self._error = error

    def __call__(self, url, headers, payload) -> bytes:
        self.calls.append({"url": url, "headers": headers, "payload": payload})
        if self._error is not None:
            raise self._error
        return _wav(self._frames)


def test_orpheus_posts_to_the_speech_endpoint_with_a_bearer_token():
    post = FakePost()
    orpheus_speech("hello", "en", api_key="sk-test", post=post)

    call = post.calls[0]
    assert call["url"] == ORPHEUS_ENDPOINT
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["payload"] == {
        "model": ORPHEUS_MODELS["en"],
        "input": "hello",
        "voice": "tara",
        "response_format": "wav",
    }


def test_orpheus_uses_the_arabic_model_for_arabic():
    post = FakePost()
    orpheus_speech("مرحبا", "ar", api_key="sk-test", post=post)
    assert post.calls[0]["payload"]["model"] == ORPHEUS_MODELS["ar"]


def test_orpheus_rejects_languages_it_does_not_cover():
    post = FakePost()
    with pytest.raises(ValueError, match="does not cover"):
        orpheus_speech("bonjour", "fr", api_key="sk-test", post=post)
    assert post.calls == [], "no request should be made for an uncovered language"


def test_orpheus_chunks_long_input_and_concatenates_the_audio():
    post = FakePost(frames=b"\x01\x02" * 8)
    text = ". ".join(f"sentence number {i}" for i in range(200)) + "."
    audio = orpheus_speech(text, "en", api_key="sk-test", post=post)

    assert len(post.calls) > 1, "long input must be chunked"
    assert "".join(c["payload"]["input"] for c in post.calls) == text
    # One WAV header, every chunk's frames present exactly once.
    assert audio.count(b"RIFF") == 1
    assert len(_frames(audio)) == len(post.calls) * 16


def test_orpheus_failure_propagates_so_the_caller_can_fall_back():
    post = FakePost(error=RuntimeError("400 model_terms_required"))
    with pytest.raises(RuntimeError, match="model_terms_required"):
        orpheus_speech("hello", "en", api_key="sk-test", post=post)


def test_make_orpheus_fn_is_wired_into_synthesize_speech():
    post = FakePost()
    audio = synthesize_speech(
        "hello",
        language="en",
        enable_orpheus=True,
        gtts_factory=FakeGTTS,
        orpheus_fn=make_orpheus_fn("sk-test", post=post),
    )
    assert audio.startswith(b"RIFF")
    assert FakeGTTS.instances == [], "gTTS must not run when Orpheus succeeds"


def test_a_live_orpheus_400_still_produces_gtts_audio():
    """The current account state: terms not accepted, so this is the live path."""
    post = FakePost(error=RuntimeError("400 model_terms_required"))
    audio = synthesize_speech(
        "hello",
        language="en",
        enable_orpheus=True,
        gtts_factory=FakeGTTS,
        orpheus_fn=make_orpheus_fn("sk-test", post=post),
    )
    assert audio.startswith(b"MP3")
