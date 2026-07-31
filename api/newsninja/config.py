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

    #: Origins permitted to call this service from a browser. Empty by default:
    #: no cross-origin access is a safe failure, "*" is not.
    allowed_origins: list[str] = []
    #: How long a request handler may sit inside the token limiter before the
    #: service answers 429 instead. Keeps a full budget from becoming a
    #: connection held open until the platform proxy severs it.
    api_max_wait_seconds: float = 5.0
    #: Per-IP request ceiling. Stops one client hammering the service; it does
    #: not protect the shared token budget — api_max_wait_seconds does that.
    rate_limit_per_minute: int = 10
    #: Read the client address from X-Forwarded-For. Off by default: the header
    #: is spoofable unless the platform overwrites it, and trusting it blindly
    #: turns the per-IP window into decoration.
    trust_proxy_headers: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Call this rather than constructing Settings."""
    return Settings()
