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


def test_a_rate_limit_of_zero_is_refused_rather_than_disabling_the_window():
    """"Set it to 0 to disable" is the natural guess, and it does the opposite.

    RateLimiter refuses at `len(hits) >= 0` and then reads hits[0] from an empty
    deque. That IndexError is raised inside the rate-limit middleware, where no
    exception handler is reachable, so every request becomes a 500 with a
    traceback. Refusing the value at load is the only place this can be caught.
    """
    with pytest.raises(ValidationError):
        Settings(rate_limit_per_minute=0)


def test_a_negative_rate_limit_is_refused():
    with pytest.raises(ValidationError):
        Settings(rate_limit_per_minute=-1)


def test_a_rate_limit_of_one_is_accepted():
    """The lowest meaningful setting must still load."""
    assert Settings(rate_limit_per_minute=1).rate_limit_per_minute == 1
