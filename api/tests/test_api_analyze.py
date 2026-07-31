import pytest
from fastapi.testclient import TestClient

from newsninja.api.deps import get_cache, get_client, get_sources
from newsninja.errors import SourceError
from newsninja.models import Article, ArticleAnalysis


class FakeSource:
    def __init__(self, name, articles=None, error=None, is_available=True):
        self.name = name
        self._articles = articles or []
        self._error = error
        self._available = is_available

    def available(self):
        return self._available

    def fetch(self, topic, limit=8):
        if self._error:
            raise self._error
        return self._articles


class FakeClient:
    def structured(self, *, model, system, user, schema_model, max_retries=2):
        return ArticleAnalysis(
            topic="ai", summary="s", entities=[], stance="neutral",
            confidence=0.5, key_claims=[],
        )

    def text(self, *, model, system, user):
        return "Here is your briefing."


def _article():
    return Article(title="T", url="u", source="google_news", body="b")


@pytest.fixture
def client(api_app):
    api_app.dependency_overrides[get_client] = FakeClient
    api_app.dependency_overrides[get_cache] = lambda: None
    api_app.dependency_overrides[get_sources] = lambda: [
        FakeSource("google_news", [_article()])
    ]
    with TestClient(api_app) as test_client:
        yield test_client


def test_analyze_returns_one_analysis(client):
    response = client.post("/analyze", json={"topic": "ai"})
    assert response.status_code == 200
    body = response.json()
    assert body["analysis"]["topic"] == "ai"
    assert body["source_errors"] == {}
    assert body["skipped_sources"] == []


def test_a_degraded_run_is_still_a_success(api_app, client):
    """A briefing built from fewer sources is a result, not a failure.

    run_pipeline already treats it that way; the HTTP layer must not disagree
    with the package it wraps.
    """
    api_app.dependency_overrides[get_sources] = lambda: [
        FakeSource("reddit", error=SourceError("reddit", "401")),
        FakeSource("google_news", [_article()]),
    ]
    response = client.post("/analyze", json={"topic": "ai"})
    assert response.status_code == 200
    assert "reddit" in response.json()["source_errors"]


def test_skipped_and_failed_sources_stay_in_separate_fields(api_app, client):
    api_app.dependency_overrides[get_sources] = lambda: [
        FakeSource("reddit", is_available=False),
        FakeSource("google_news", [_article()]),
    ]
    body = client.post("/analyze", json={"topic": "ai"}).json()
    assert body["skipped_sources"] == ["reddit"]
    assert body["source_errors"] == {}


@pytest.mark.parametrize(
    "payload",
    [
        {"topic": "   "},
        {"topic": ""},
        {"topic": "x" * 201},
        {"topic": "ai", "limit": 0},
        {"topic": "ai", "limit": 13},
    ],
)
def test_invalid_requests_are_refused(client, payload):
    assert client.post("/analyze", json=payload).status_code == 422
