import html

import pytest
from fastapi.testclient import TestClient

from evals.corpus import load_corpus
from newsninja.analysis.client import MODEL_TPM, GroqClient
from newsninja.analysis.limiter import TokenBudgetLimiter
from newsninja.analysis.synthesize import (
    SYNTHESIS_MODEL,
    TRANSLATION_MODEL,
    build_briefing,
)
from newsninja.api.deps import get_client
from newsninja.api.schemas import MAX_BRIEF_CHARS, AnalyzeResponse, BriefRequest
from newsninja.audio.tts import SUPPORTED_LANGUAGES
from newsninja.models import ArticleAnalysis
from newsninja.pipeline import MAX_TOPICS


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


def _realistic_five_topic_analyses() -> list[dict]:
    """One analysis per corpus topic, built to look like real model output.

    ``key_claims`` carry verbatim quotes copied straight out of the real
    article bodies in evals/data/corpus.jsonl, because that is what
    ``_text_chars`` counts and what a genuine extraction returns -- a claim
    whose quote is invented text would not be measuring the same thing the
    server actually holds in memory. Four claims per topic and a 2-4 sentence
    summary is what the extract.md prompt asks for; this is not the maximum
    the prompt allows, just a plausible five-topic run.
    """
    analyses = []
    for record in load_corpus():
        topic = record.topic
        articles = record.articles[:4]
        claims = [
            {
                "text": (
                    f"Reporting from {article.source} covers a development in "
                    f"{topic}: {article.title[:80]}."
                ),
                "quote": html.unescape(article.body),
            }
            for article in articles
        ]
        summary = (
            f"Recent coverage of {topic} spans {len(record.articles)} sources this "
            f'cycle. Notable developments include "{articles[0].title[:60]}" and '
            f'"{articles[1].title[:60]}". Sentiment is mixed, with both '
            "opportunity and risk emphasized across outlets, and no single source "
            "dominating the narrative."
        )
        analyses.append(
            {
                "topic": topic,
                "summary": summary,
                "entities": [
                    {"name": name, "kind": "org"}
                    for name in ["Reuters", "Bloomberg", "AP"]
                ],
                "stance": "neutral",
                "confidence": 0.72,
                "key_claims": claims,
            }
        )
    return analyses


def test_a_realistic_five_topic_brief_request_is_accepted(client):
    """The documented workflow is N x /analyze then one /brief; this is that
    request for a genuine five-topic run, not a synthetic worst case.

    Built from evals/data/corpus.jsonl rather than asserted: this measures
    8,330 characters of text, which the old 8,000-character MAX_BRIEF_CHARS
    would have rejected with a 422 on a request the documented workflow
    produces legitimately.
    """
    payload = {"analyses": _realistic_five_topic_analyses()}

    response = client.post("/brief", json=payload)

    assert response.status_code == 200
    assert response.json()["briefing"]["topics"] == [
        "artificial intelligence",
        "climate change",
        "cryptocurrency",
        "space exploration",
        "renewable energy",
    ]


class RecordingClient:
    """Records which models were called, which is how the language shows up."""

    def __init__(self):
        self.models = []

    def structured(self, *, model, system, user, schema_model, max_retries=2):
        raise AssertionError("/brief must not run extraction")

    def text(self, *, model, system, user):
        self.models.append(model)
        return "Here is your briefing."


def test_the_requested_language_reaches_the_briefing(api_app, client):
    """Verified: forcing language="en" in the handler left the suite green.

    Only rejected codes were covered. A good code was never followed through to
    build_briefing or back out in the response.
    """
    recorder = RecordingClient()
    api_app.dependency_overrides[get_client] = lambda: recorder

    payload = {"analyses": [_analysis()], "language": "fr"}
    response = client.post("/brief", json=payload)

    assert response.status_code == 200
    assert response.json()["briefing"]["language"] == "fr"
    # Synthesis, then translation. translate() is a no-op for English, so a
    # handler that forced "en" would make one call rather than two.
    assert recorder.models == [SYNTHESIS_MODEL, TRANSLATION_MODEL]


def test_exactly_five_analyses_are_accepted(client):
    """The boundary itself, not just the two sides of it.

    Zero and six are covered. If max_length were MAX_TOPICS - 1 both of those
    tests would still pass while every legitimate five-topic briefing — the
    thing the endpoint exists for — came back 422.
    """
    payload = {"analyses": [_analysis(f"t{n}") for n in range(MAX_TOPICS)]}

    response = client.post("/brief", json=payload)

    assert response.status_code == 200
    assert response.json()["briefing"]["topics"] == [
        f"t{n}" for n in range(MAX_TOPICS)
    ]


@pytest.mark.parametrize("language", ["sv", "nl", "pl", "en-GB"])
def test_a_language_the_pipeline_cannot_speak_is_refused(client, language):
    """Shape-valid but unsupported.

    synthesize_speech rewrites anything outside SUPPORTED_LANGUAGES to "en"
    without saying so, so accepting "sv" means returning a Swedish script that
    /audio then speaks in English, with nothing in either response saying it
    happened. 422 is the only answer that does not mislead.
    """
    payload = {"analyses": [_analysis()], "language": language}
    assert client.post("/brief", json=payload).status_code == 422


@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
def test_every_supported_language_is_accepted(client, language):
    """The refusal must not overshoot what the pipeline can actually do."""
    payload = {"analyses": [_analysis()], "language": language}
    response = client.post("/brief", json=payload)
    assert response.status_code == 200
    assert response.json()["briefing"]["language"] == language


def test_an_analyze_response_is_not_itself_a_brief_analysis(client):
    """The caller plucks `.analysis`; the response object whole is a 422.

    The spec used to say `analyses` holds objects "in the same shape /analyze
    returns", which would have sent a reader straight into a 422. Built from
    AnalyzeResponse rather than a literal so it cannot drift from that schema.
    """
    whole = AnalyzeResponse(
        analysis=ArticleAnalysis.model_validate(_analysis())
    ).model_dump(mode="json")

    assert client.post("/brief", json={"analyses": [whole]}).status_code == 422
    plucked = {"analyses": [whole["analysis"]]}
    assert client.post("/brief", json=plucked).status_code == 200
