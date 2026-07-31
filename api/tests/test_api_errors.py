import pytest
from fastapi.testclient import TestClient

from newsninja.api.app import create_app
from newsninja.api.deps import get_cache, get_client, get_sources
from newsninja.config import Settings, get_settings
from newsninja.errors import ExtractionFailure, RateLimitError, SourceError
from newsninja.models import Article


def _sources():
    return [_OneArticleSource()]


class _OneArticleSource:
    name = "google_news"

    def available(self):
        return True

    def fetch(self, topic, limit=8):
        return [Article(title="T", url="u", source="google_news", body="b")]


def _raising_client(exc):
    class Raising:
        def structured(self, *, model, system, user, schema_model, max_retries=2):
            raise exc

        def text(self, *, model, system, user):
            raise exc

    return Raising


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """get_settings is lru_cached, so a monkeypatched variable stays invisible
    until the cache is dropped — before the test as well as after it."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(api_app):
    api_app.dependency_overrides[get_cache] = lambda: None
    api_app.dependency_overrides[get_sources] = _sources
    with TestClient(api_app) as test_client:
        yield test_client


def test_a_rate_limit_becomes_429_with_a_retry_after_header(api_app, client):
    """The header is what proxies and browsers honour; the body is what the
    frontend can render without reading headers through CORS. Both, not one."""
    api_app.dependency_overrides[get_client] = _raising_client(
        RateLimitError("budget full", retry_after=48.0)
    )
    response = client.post("/analyze", json={"topic": "ai"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "48"
    body = response.json()["error"]
    assert body["type"] == "rate_limit"
    assert body["retry_after"] == 48.0


def test_retry_after_rounds_up_rather_than_truncating(api_app, client):
    """Truncating 0.4s to "0" tells the caller to retry immediately into a
    budget that is still full."""
    api_app.dependency_overrides[get_client] = _raising_client(
        RateLimitError("budget full", retry_after=0.4)
    )
    response = client.post("/analyze", json={"topic": "ai"})
    assert response.headers["retry-after"] == "1"


def test_an_extraction_failure_becomes_502(api_app, client):
    api_app.dependency_overrides[get_client] = _raising_client(
        ExtractionFailure("did not validate")
    )
    response = client.post("/analyze", json={"topic": "ai"})
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "extraction_failure"


def test_a_source_error_becomes_502_and_names_the_source(api_app, client):
    api_app.dependency_overrides[get_client] = _raising_client(
        SourceError("reddit", "401")
    )
    response = client.post("/analyze", json={"topic": "ai"})
    assert response.status_code == 502
    assert "reddit" in response.json()["error"]["message"]


def test_the_rate_limit_window_refuses_a_hammering_client():
    """The window counts requests, not successes.

    An empty body is used deliberately: it is refused at validation, so the
    handler never runs and no source is ever fetched. A payload that reached
    the real sources would put this test on the network.
    """
    limited = create_app(Settings(rate_limit_per_minute=2))
    with TestClient(limited) as test_client:
        assert test_client.post("/analyze", json={}).status_code == 422
        assert test_client.post("/analyze", json={}).status_code == 422
        response = test_client.post("/analyze", json={})
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit"


def test_health_is_exempt_from_the_window():
    """Uptime pings must not consume a visitor's allowance."""
    limited = create_app(Settings(rate_limit_per_minute=1))
    with TestClient(limited) as test_client:
        codes = [test_client.get("/health").status_code for _ in range(5)]
    assert codes == [200] * 5


def test_retry_after_is_readable_by_a_browser():
    """CORS hides all but a six-header safelist from JavaScript, and
    Retry-After is not on it. Without expose_headers the frontend receives the
    429 and cannot read how long to wait."""
    origin = "https://example.vercel.app"
    cors = create_app(Settings(allowed_origins=[origin]))
    with TestClient(cors) as test_client:
        response = test_client.get("/health", headers={"Origin": origin})
    assert "Retry-After" in response.headers["access-control-expose-headers"]


def test_an_unlisted_origin_is_not_granted_access():
    """An empty allowlist must mean no origin, not any origin."""
    cors = create_app(Settings(allowed_origins=["https://example.vercel.app"]))
    with TestClient(cors) as test_client:
        response = test_client.get("/health", headers={"Origin": "https://evil.test"})
    assert "access-control-allow-origin" not in response.headers
