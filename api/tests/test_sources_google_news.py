from pathlib import Path

import pytest

from newsninja.errors import SourceError
from newsninja.sources.google_news import GoogleNewsSource

FIXTURE = Path(__file__).parent / "fixtures" / "google_news_ai.xml"


def _source() -> GoogleNewsSource:
    return GoogleNewsSource(fetcher=lambda url, timeout: FIXTURE.read_bytes())


def test_is_always_available():
    assert _source().available() is True


def test_fetch_returns_articles():
    articles = _source().fetch("artificial intelligence")
    assert articles
    assert all(a.source == "google_news" for a in articles)
    assert all(a.title for a in articles)
    assert all(a.url.startswith("http") for a in articles)


def test_fetch_respects_the_limit():
    assert len(_source().fetch("artificial intelligence", limit=3)) == 3


def test_titles_have_html_stripped():
    for article in _source().fetch("artificial intelligence"):
        assert "<" not in article.title


def test_transport_failure_raises_source_error():
    def boom(url, timeout):
        raise OSError("network down")

    with pytest.raises(SourceError) as excinfo:
        GoogleNewsSource(fetcher=boom).fetch("ai")
    assert excinfo.value.source == "google_news"
