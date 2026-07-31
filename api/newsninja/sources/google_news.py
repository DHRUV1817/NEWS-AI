"""Google News RSS. Free, no authentication. Verified reachable 2026-07-31."""

import re
from collections.abc import Callable
from urllib.parse import quote_plus

import feedparser

from newsninja.errors import SourceError
from newsninja.models import Article

_TAG = re.compile(r"<[^>]+>")
_USER_AGENT = "Mozilla/5.0 (compatible; NewsNinja/3.0; +https://github.com/DHRUV1817)"


def _default_fetcher(url: str, timeout: int) -> bytes:
    import httpx

    response = httpx.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.content


class GoogleNewsSource:
    name = "google_news"

    def __init__(
        self,
        fetcher: Callable[[str, int], bytes] = _default_fetcher,
        timeout: int = 30,
    ) -> None:
        self._fetch_bytes = fetcher
        self._timeout = timeout

    def available(self) -> bool:
        return True

    def fetch(self, topic: str, limit: int = 8) -> list[Article]:
        url = (
            "https://news.google.com/rss/search"
            f"?q={quote_plus(topic)}&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            raw = self._fetch_bytes(url, self._timeout)
        except Exception as exc:
            raise SourceError(self.name, str(exc)) from exc

        feed = feedparser.parse(raw)
        articles: list[Article] = []
        for entry in feed.entries[:limit]:
            title = _TAG.sub("", getattr(entry, "title", "")).strip()
            if not title:
                continue
            summary = _TAG.sub("", getattr(entry, "summary", "")).strip()
            articles.append(
                Article(
                    title=title,
                    url=getattr(entry, "link", ""),
                    source=self.name,
                    body=f"{title}. {summary}".strip(),
                )
            )
        return articles
