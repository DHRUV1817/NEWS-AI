import pytest
from fastapi.testclient import TestClient

from newsninja.api.deps import get_client


@pytest.fixture
def client(api_app):
    with TestClient(api_app) as test_client:
        yield test_client


def test_health_reports_ok_and_the_package_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_health_never_calls_the_model(client, monkeypatch):
    """/health must not construct or call the model client.

    Patching the function rather than registering a FastAPI dependency
    override is what makes this discriminate: /health declares no
    dependencies, so an override is never consulted, and the test would pass
    even if the handler called get_client() directly.
    """
    from newsninja.api import deps, routes

    def _explode(*args, **kwargs):
        raise AssertionError("/health must not build or call the model client")

    monkeypatch.setattr(deps, "get_client", _explode)
    monkeypatch.setattr(routes, "get_client", _explode, raising=False)

    assert client.get("/health").status_code == 200


def test_the_model_client_is_one_instance_for_the_process():
    """Silent when broken, so it is asserted directly.

    GroqClient owns the TokenBudgetLimiter. A client built per request would
    hand every request a fresh, empty budget window, the limiter would never
    wait, and the first symptom would be Groq 429s in production.
    """
    get_client.cache_clear()
    assert get_client() is get_client()
    get_client.cache_clear()
