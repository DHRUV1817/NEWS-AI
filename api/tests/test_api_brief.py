import pytest
from fastapi.testclient import TestClient

from newsninja.api.deps import get_client


class FakeClient:
    def structured(self, *, model, system, user, schema_model, max_retries=2):
        raise AssertionError("/brief must not run extraction")

    def text(self, *, model, system, user):
        return "Here is your briefing."


def _analysis(topic="ai"):
    return {
        "topic": topic,
        "summary": "s",
        "entities": [],
        "stance": "neutral",
        "confidence": 0.5,
        "key_claims": [],
    }


@pytest.fixture
def client(api_app):
    api_app.dependency_overrides[get_client] = FakeClient
    with TestClient(api_app) as test_client:
        yield test_client


def test_brief_synthesises_across_every_analysis(client):
    """The unified script is the product. Synthesising per batch would yield
    two disconnected briefings for five topics, which is the reason this
    endpoint takes all the analyses at once."""
    payload = {"analyses": [_analysis("ai"), _analysis("climate")], "language": "en"}
    response = client.post("/brief", json=payload)
    assert response.status_code == 200
    briefing = response.json()["briefing"]
    assert briefing["script"] == "Here is your briefing."
    assert briefing["topics"] == ["ai", "climate"]


def test_more_than_five_analyses_are_refused(client):
    payload = {"analyses": [_analysis(f"t{n}") for n in range(6)]}
    assert client.post("/brief", json=payload).status_code == 422


def test_no_analyses_is_refused(client):
    assert client.post("/brief", json={"analyses": []}).status_code == 422


@pytest.mark.parametrize("language", ["english", "e", "EN", "en_GB", "'; DROP"])
def test_a_language_code_is_not_free_text(client, language):
    """It reaches a translation prompt, so it is validated rather than trusted."""
    payload = {"analyses": [_analysis()], "language": language}
    assert client.post("/brief", json=payload).status_code == 422
