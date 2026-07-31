from typing import ClassVar

import pytest

from newsninja.audio.tts import (
    ORPHEUS_LANGUAGES,
    SUPPORTED_LANGUAGES,
    chunk_text,
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
