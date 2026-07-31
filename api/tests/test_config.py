from pathlib import Path

import pytest
from pydantic import ValidationError

from newsninja.config import Settings


def test_settings_reads_groq_key_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_example")
    settings = Settings(_env_file=None)
    assert settings.groq_api_key == "gsk_example"


def test_groq_key_is_required(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_reddit_credentials_default_to_none(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_example")
    settings = Settings(_env_file=None)
    assert settings.reddit_client_id is None
    assert settings.reddit_client_secret is None


def test_orpheus_is_off_by_default(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_example")
    assert Settings(_env_file=None).enable_orpheus is False


def test_cache_path_is_a_path(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_example")
    assert isinstance(Settings(_env_file=None).cache_path, Path)


ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def _documented_keys() -> set[str]:
    """Every setting named in .env.example, whether live or commented out."""
    keys = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().lstrip("#").strip()
        name, sep, _ = stripped.partition("=")
        if sep and name.isupper() and name.replace("_", "").isalnum():
            keys.add(name)
    return keys


def test_env_example_documents_exactly_the_real_settings():
    """The stale template described a deleted implementation and omitted
    GROQ_API_KEY, so following the README's setup step produced an unusable
    config. Keep it mechanically in step with Settings."""
    expected = {name.upper() for name in Settings.model_fields}
    assert _documented_keys() == expected


def test_service_settings_have_safe_defaults():
    """Defaults must be safe to deploy without reading the docs first.

    No origins means no cross-origin access rather than any; not trusting proxy
    headers means a spoofed X-Forwarded-For cannot bypass the per-IP window.
    """
    settings = Settings()
    assert settings.allowed_origins == []
    assert settings.trust_proxy_headers is False
    assert settings.api_max_wait_seconds == 5.0
    assert settings.rate_limit_per_minute == 10


def test_allowed_origins_reads_a_json_list_from_the_environment(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", '["https://example.vercel.app"]')
    assert Settings().allowed_origins == ["https://example.vercel.app"]
