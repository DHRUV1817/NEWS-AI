import pytest
from fastapi.testclient import TestClient

from newsninja.api.deps import get_tts
from newsninja.api.schemas import MAX_SCRIPT_CHARS
from newsninja.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """get_settings is lru_cached, so a monkeypatched variable is invisible
    until the cache is dropped — before the test as well as after it."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(api_app):
    api_app.dependency_overrides[get_tts] = lambda: (
        lambda text, language, enable_orpheus: b"AUDIO"
    )
    with TestClient(api_app) as test_client:
        yield test_client


def test_audio_returns_bytes_not_json(client):
    response = client.post("/audio", json={"script": "hello", "language": "en"})
    assert response.status_code == 200
    assert response.content == b"AUDIO"
    assert response.headers["content-type"] == "audio/mpeg"


def test_the_content_type_follows_the_configured_engine(client, monkeypatch):
    """gTTS returns mp3 and Orpheus returns wav, so the header cannot be a
    constant without lying about one of them."""
    monkeypatch.setenv("ENABLE_ORPHEUS", "true")
    get_settings.cache_clear()
    response = client.post("/audio", json={"script": "hello"})
    assert response.headers["content-type"] == "audio/wav"


def test_an_oversized_script_is_refused(client):
    """chunk_text splits at 3,000 characters and gTTS makes one outbound call
    per chunk, so an uncapped script is an amplification vector."""
    payload = {"script": "x" * (MAX_SCRIPT_CHARS + 1)}
    assert client.post("/audio", json=payload).status_code == 422


def test_an_empty_script_is_refused(client):
    assert client.post("/audio", json={"script": ""}).status_code == 422
