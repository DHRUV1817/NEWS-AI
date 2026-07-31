import pytest
from fastapi.testclient import TestClient

from newsninja.api.deps import get_tts
from newsninja.api.schemas import MAX_SCRIPT_CHARS
from newsninja.audio.tts import MP3_MEDIA_TYPE, WAV_MEDIA_TYPE, SpokenAudio
from newsninja.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """get_settings is lru_cached, so a monkeypatched variable is invisible
    until the cache is dropped — before the test as well as after it."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _speaking(media_type):
    """A speech seam that reports the format it produced, like the real one."""
    return lambda: (
        lambda text, language, enable_orpheus: SpokenAudio(b"AUDIO", media_type)
    )


@pytest.fixture
def client(api_app):
    api_app.dependency_overrides[get_tts] = _speaking(MP3_MEDIA_TYPE)
    with TestClient(api_app) as test_client:
        yield test_client


def test_audio_returns_bytes_not_json(client):
    response = client.post("/audio", json={"script": "hello", "language": "en"})
    assert response.status_code == 200
    assert response.content == b"AUDIO"
    assert response.headers["content-type"] == "audio/mpeg"


def test_the_content_type_follows_the_format_actually_produced(
    api_app, client, monkeypatch
):
    """gTTS returns mp3 and Orpheus returns wav, so the header cannot be a
    constant without lying about one of them — and it cannot be read off
    ENABLE_ORPHEUS either, because that says what was asked for."""
    api_app.dependency_overrides[get_tts] = _speaking(WAV_MEDIA_TYPE)
    monkeypatch.setenv("ENABLE_ORPHEUS", "true")
    get_settings.cache_clear()
    response = client.post("/audio", json={"script": "hello"})
    assert response.headers["content-type"] == "audio/wav"


def test_the_content_type_follows_a_silent_fallback_to_gtts(
    api_app, client, monkeypatch
):
    """The live path on an account that has not accepted the Orpheus terms.

    Every Orpheus failure falls back to gTTS, which returns mp3. Reading the
    header off ENABLE_ORPHEUS served mp3 bytes labelled audio/wav.
    """
    api_app.dependency_overrides[get_tts] = _speaking(MP3_MEDIA_TYPE)
    monkeypatch.setenv("ENABLE_ORPHEUS", "true")
    get_settings.cache_clear()
    response = client.post("/audio", json={"script": "hello"})
    assert response.headers["content-type"] == "audio/mpeg"


def test_an_oversized_script_is_refused(client):
    """chunk_text splits at 3,000 characters and gTTS makes one outbound call
    per chunk, so an uncapped script is an amplification vector."""
    payload = {"script": "x" * (MAX_SCRIPT_CHARS + 1)}
    assert client.post("/audio", json=payload).status_code == 422


def test_an_empty_script_is_refused(client):
    assert client.post("/audio", json={"script": ""}).status_code == 422


def test_the_live_seam_reports_mp3_when_orpheus_is_enabled_but_fails():
    """End to end through the real seam, no dependency override.

    This is the assertion that would have caught the original bug: the route's
    header comes from render_speech, and render_speech knows that the Orpheus
    call raised and gTTS produced the bytes.
    """
    from newsninja.audio.tts import render_speech

    def _failing_orpheus(text, language):
        raise RuntimeError("400 model_terms_required")

    class _FakeGTTS:
        def __init__(self, text, lang, slow=False):
            self.text = text

        def write_to_fp(self, fp):
            fp.write(b"MP3")

    spoken = render_speech(
        "hello",
        language="en",
        enable_orpheus=True,
        gtts_factory=_FakeGTTS,
        orpheus_fn=_failing_orpheus,
    )
    assert spoken.data.startswith(b"MP3")
    assert spoken.media_type == MP3_MEDIA_TYPE
