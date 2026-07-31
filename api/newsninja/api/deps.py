"""Process-wide singletons, supplied to handlers as FastAPI dependencies.

``lru_cache`` is what makes them singletons, and that is load-bearing rather
than an optimisation. ``GroqClient`` owns the ``TokenBudgetLimiter``; building
one per request would give every request a fresh, empty budget window, so the
limiter would never wait and the service would walk straight into Groq 429s
with no local warning at all.
"""

from collections.abc import Callable
from functools import lru_cache

from newsninja.analysis.client import GroqClient
from newsninja.cache import Cache
from newsninja.config import get_settings
from newsninja.pipeline import default_tts
from newsninja.sources.base import Source
from newsninja.sources.google_news import GoogleNewsSource
from newsninja.sources.reddit import RedditSource


@lru_cache(maxsize=1)
def get_client() -> GroqClient:
    settings = get_settings()
    return GroqClient(
        api_key=settings.groq_api_key,
        max_wait=settings.api_max_wait_seconds,
    )


@lru_cache(maxsize=1)
def get_cache() -> Cache:
    return Cache(get_settings().cache_path)


@lru_cache(maxsize=1)
def get_sources() -> list[Source]:
    settings = get_settings()
    return [
        GoogleNewsSource(timeout=settings.request_timeout),
        RedditSource(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        ),
    ]


@lru_cache(maxsize=1)
def get_tts() -> Callable[[str, str, bool], bytes]:
    # Orpheus is a separate REST endpoint rather than part of the chat client,
    # so the key has to reach the speech seam too.
    return default_tts(get_settings().groq_api_key)
