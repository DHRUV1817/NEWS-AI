import json

from newsninja.analysis.extract import extract_topic
from newsninja.cache import Cache
from newsninja.models import Article, ArticleAnalysis

VALID = {
    "topic": "ai",
    "summary": "Models improved.",
    "entities": [{"name": "OpenAI", "kind": "org"}],
    "stance": "positive",
    "confidence": 0.8,
    "key_claims": [{"text": "Models improved", "quote": "models improved this year"}],
}


class RecordingClient:
    def __init__(self, payload=None):
        self.payload = payload or VALID
        self.calls = []

    def structured(self, *, model, system, user, schema_model, max_retries=2):
        self.calls.append({"model": model, "system": system, "user": user})
        return schema_model.model_validate(self.payload)


def _articles():
    return [
        Article(title="A", url="u1", source="google_news", body="models improved this year"),
        Article(title="B", url="u2", source="reddit", body="people are excited"),
    ]


def test_returns_an_analysis():
    result = extract_topic(RecordingClient(), "ai", _articles())
    assert isinstance(result, ArticleAnalysis)
    assert result.stance == "positive"


def test_all_articles_go_in_one_call():
    client = RecordingClient()
    extract_topic(client, "ai", _articles())
    assert len(client.calls) == 1, "articles must be batched into a single request"
    assert "models improved this year" in client.calls[0]["user"]
    assert "people are excited" in client.calls[0]["user"]


def test_uses_a_schema_capable_model():
    client = RecordingClient()
    extract_topic(client, "ai", _articles())
    assert client.calls[0]["model"] == "openai/gpt-oss-20b"


def test_empty_articles_returns_a_neutral_analysis_without_calling_the_model():
    client = RecordingClient()
    result = extract_topic(client, "ai", [])
    assert client.calls == []
    assert result.stance == "neutral"
    assert result.key_claims == []


def test_result_is_cached(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    client = RecordingClient()
    extract_topic(client, "ai", _articles(), cache=cache)
    extract_topic(client, "ai", _articles(), cache=cache)
    assert len(client.calls) == 1, "second call must be served from cache"


def test_cached_value_round_trips(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    first = extract_topic(RecordingClient(), "ai", _articles(), cache=cache)
    second = extract_topic(RecordingClient(), "ai", _articles(), cache=cache)
    assert first == second
