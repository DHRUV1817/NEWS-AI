import pytest
from fastapi.testclient import TestClient

from newsninja.analysis.client import MODEL_TPM, GroqClient
from newsninja.analysis.limiter import TokenBudgetLimiter
from newsninja.analysis.synthesize import SYNTHESIS_MODEL, build_briefing
from newsninja.api.deps import get_client
from newsninja.api.schemas import MAX_BRIEF_CHARS, BriefRequest


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


def _analysis_of_size(chars: int, topic="ai"):
    """An analysis whose counted free text is exactly ``chars`` characters."""
    analysis = _analysis(topic)
    analysis["summary"] = "s" * (chars - len(topic))
    return analysis


def test_a_brief_request_at_the_character_bound_is_accepted(client):
    payload = {"analyses": [_analysis_of_size(MAX_BRIEF_CHARS)]}
    assert client.post("/brief", json=payload).status_code == 200


def test_an_oversized_brief_request_is_refused(client):
    """Bounding the count without bounding the size bounds nothing.

    Four analyses with 6,000-character summaries reserve 7,737 of gpt-oss-120b's
    8,000 TPM — one unauthenticated request holding 97% of the shared minute.
    """
    payload = {"analyses": [_analysis_of_size(MAX_BRIEF_CHARS + 1)]}
    assert client.post("/brief", json=payload).status_code == 422


def test_the_character_bound_is_over_the_request_not_one_analysis(client):
    """Five analyses each under the bound must not sum past it."""
    payload = {
        "analyses": [
            _analysis_of_size(MAX_BRIEF_CHARS // 4, topic=f"t{n}") for n in range(5)
        ]
    }
    assert client.post("/brief", json=payload).status_code == 422


def test_a_maximal_brief_request_reserves_under_half_the_budget():
    """The arithmetic MAX_BRIEF_CHARS is justified by, asserted rather than believed.

    The transport stops the call the moment it is reached, so what the limiter
    holds afterwards is the reservation itself rather than a settled usage.
    """

    class _Reached(Exception):
        pass

    class _StoppingTransport:
        def complete(self, **kwargs):
            raise _Reached

    tpm = MODEL_TPM[SYNTHESIS_MODEL]
    limiter = TokenBudgetLimiter(tpm=tpm)
    groq = GroqClient(
        api_key="test-key-not-real",
        limiters={SYNTHESIS_MODEL: limiter},
        transport=_StoppingTransport(),
    )
    biggest = BriefRequest.model_validate(
        {"analyses": [_analysis_of_size(MAX_BRIEF_CHARS)]}
    )

    with pytest.raises(_Reached):
        build_briefing(groq, biggest.analyses)

    assert limiter.used_tokens() < tpm // 2, (
        f"a maximal /brief request reserved {limiter.used_tokens()} of {tpm}"
    )
