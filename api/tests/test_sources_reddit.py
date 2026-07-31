import pytest

from newsninja.errors import SourceError
from newsninja.sources.reddit import RedditSource


class FakeSubmission:
    def __init__(self, title, score, num_comments, permalink, selftext=""):
        self.title = title
        self.score = score
        self.num_comments = num_comments
        self.permalink = permalink
        self.selftext = selftext


class FakeReddit:
    def __init__(self, submissions):
        self._submissions = submissions

    def subreddit(self, name):
        return self

    def search(self, query, sort="hot", time_filter="week", limit=8):
        return iter(self._submissions[:limit])


def _source(submissions=None):
    submissions = submissions if submissions is not None else [
        FakeSubmission("AI breakthrough announced", 420, 87, "/r/tech/1"),
        FakeSubmission("Thoughts on the new model", 88, 12, "/r/tech/2"),
    ]
    return RedditSource(
        client_id="id",
        client_secret="secret",
        user_agent="test",
        client_factory=lambda **kw: FakeReddit(submissions),
    )


def test_unavailable_without_credentials():
    source = RedditSource(client_id=None, client_secret=None, user_agent="test")
    assert source.available() is False


def test_available_with_credentials():
    assert _source().available() is True


def test_fetch_without_credentials_raises():
    source = RedditSource(client_id=None, client_secret=None, user_agent="test")
    with pytest.raises(SourceError):
        source.fetch("ai")


def test_fetch_returns_articles_with_engagement_in_the_body():
    articles = _source().fetch("ai")
    assert len(articles) == 2
    assert articles[0].source == "reddit"
    assert "420" in articles[0].body
    assert articles[0].url.startswith("https://www.reddit.com")


def test_api_failure_raises_source_error():
    class Exploding:
        def subreddit(self, name):
            raise RuntimeError("401 unauthorized")

    source = RedditSource(
        client_id="id",
        client_secret="secret",
        user_agent="test",
        client_factory=lambda **kw: Exploding(),
    )
    with pytest.raises(SourceError) as excinfo:
        source.fetch("ai")
    assert excinfo.value.source == "reddit"
