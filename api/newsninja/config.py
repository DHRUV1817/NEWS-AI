"""Configuration loaded from environment or .env. Single source of truth."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Only ``groq_api_key`` is required. Reddit credentials are optional: without
    them the Reddit source reports itself unavailable rather than failing.
    """

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str

    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str = "newsninja/3.0 (portfolio project)"

    enable_orpheus: bool = False

    cache_path: Path = Path(".cache/newsninja.sqlite")
    request_timeout: int = 30


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Call this rather than constructing Settings."""
    return Settings()
