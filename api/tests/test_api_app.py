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


def test_health_never_calls_the_model(api_app, client, monkeypatch):
    """/health must not construct or call the model client.

    Both mechanisms are needed and neither is redundant. A dependency
    override is consulted per request and catches a Depends(get_client)
    added to the signature, but never fires for a handler that declares no
    dependencies. Patching the module attribute catches a direct call in the
    handler body, but cannot rewire a Depends already bound to the function
    object at import. Each covers the shape the other misses.
    """
    from newsninja.api import deps, routes

    def _explode() -> object:
        raise AssertionError("/health must not build or call the model client")

    api_app.dependency_overrides[get_client] = _explode
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
