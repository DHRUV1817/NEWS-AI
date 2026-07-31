from newsninja.analysis.synthesize import build_briefing, synthesize, translate
from newsninja.models import ArticleAnalysis, Briefing, Claim, Entity


def _analysis(topic="ai", stance="positive"):
    return ArticleAnalysis(
        topic=topic,
        summary=f"Summary about {topic}.",
        entities=[Entity(name="OpenAI", kind="org")],
        stance=stance,
        confidence=0.8,
        key_claims=[Claim(text="c", quote="q")],
    )


class StubClient:
    def __init__(self, reply="Good evening. Here is your briefing."):
        self.reply = reply
        self.calls = []

    def text(self, *, model, system, user):
        self.calls.append({"model": model, "system": system, "user": user})
        return self.reply

    def structured(self, **kwargs):
        raise AssertionError("synthesis must use free-form text, not a schema call")


def test_synthesize_returns_a_script():
    client = StubClient()
    script = synthesize(client, [_analysis()])
    assert script == "Good evening. Here is your briefing."


def test_synthesize_uses_the_reasoning_model():
    client = StubClient()
    synthesize(client, [_analysis()])
    assert client.calls[0]["model"] == "openai/gpt-oss-120b"


def test_synthesize_includes_every_topic():
    client = StubClient()
    synthesize(client, [_analysis("ai"), _analysis("climate")])
    user = client.calls[0]["user"]
    assert "ai" in user and "climate" in user


def test_translate_uses_the_high_budget_model():
    client = StubClient(reply="Bonjour.")
    translate(client, "Hello.", "fr")
    assert client.calls[0]["model"] == "llama-3.3-70b-versatile"
    assert "fr" in client.calls[0]["user"] or "fr" in client.calls[0]["system"]


def test_translate_to_english_is_a_no_op():
    client = StubClient()
    assert translate(client, "Hello.", "en") == "Hello."
    assert client.calls == [], "English needs no translation call"


def test_build_briefing_returns_a_briefing():
    briefing = build_briefing(StubClient(), [_analysis()])
    assert isinstance(briefing, Briefing)
    assert briefing.topics == ["ai"]
    assert briefing.language == "en"


def test_build_briefing_translates_when_asked():
    client = StubClient(reply="translated")
    briefing = build_briefing(client, [_analysis()], language="fr")
    assert briefing.language == "fr"
    assert briefing.script == "translated"
