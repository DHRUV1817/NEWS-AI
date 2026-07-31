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
