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


def test_health_never_calls_the_model(api_app, client):
    """A health check that spends tokens against an 8,000 TPM budget causes
    the outages it is supposed to detect."""

    def _explode():
        raise AssertionError("/health must not build or call the model client")

    api_app.dependency_overrides[get_client] = _explode
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
