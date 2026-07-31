import pytest

from newsninja.errors import SourceError
from newsninja.models import Article, ArticleAnalysis
from newsninja.pipeline import run_pipeline


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


def _article(source="google_news"):
    return Article(title="T", url="u", source=source, body="b")


def _tts(text, language, enable_orpheus):
    return b"AUDIO"


def test_produces_a_briefing_and_audio():
    result = run_pipeline(
        topics=["ai"],
        sources=[FakeSource("google_news", [_article()])],
        client=FakeClient(),
        tts=_tts,
    )
    assert result.briefing.script == "Here is your briefing."
    assert result.audio == b"AUDIO"
    assert result.source_errors == {}


def test_a_failing_source_does_not_stop_the_others():
    result = run_pipeline(
        topics=["ai"],
        sources=[
            FakeSource("reddit", error=SourceError("reddit", "401")),
            FakeSource("google_news", [_article()]),
        ],
        client=FakeClient(),
        tts=_tts,
    )
    assert "reddit" in result.source_errors
    assert result.briefing.analyses


def test_unavailable_sources_are_skipped_without_error():
    result = run_pipeline(
        topics=["ai"],
        sources=[
            FakeSource("reddit", is_available=False),
            FakeSource("google_news", [_article()]),
        ],
        client=FakeClient(),
        tts=_tts,
    )
    assert result.source_errors == {}


def test_empty_topics_raises():
    with pytest.raises(ValueError):
        run_pipeline(topics=[], sources=[], client=FakeClient(), tts=_tts)


def test_more_than_five_topics_raises():
    with pytest.raises(ValueError, match="at most 5"):
        run_pipeline(
            topics=["a", "b", "c", "d", "e", "f"],
            sources=[FakeSource("google_news", [_article()])],
            client=FakeClient(),
            tts=_tts,
        )


def test_cli_reports_missing_credentials_without_traceback(capsys, monkeypatch):
    """The CLI must fail with a readable message, not a traceback.

    `Settings` normally reads ../.env, which on a developer machine holds a real
    key — so the class is swapped for one with env_file disabled. `cli.main`
    imports Settings inside the function body, so patching the module attribute
    takes effect at call time.
    """
    from pydantic_settings import SettingsConfigDict

    import newsninja.config as config_module
    from newsninja.cli import main

    class NoEnvFileSettings(config_module.Settings):
        model_config = SettingsConfigDict(env_file=None, extra="ignore")

    monkeypatch.setattr(config_module, "Settings", NoEnvFileSettings)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    exit_code = main(["--topic", "ai", "--no-audio"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "GROQ_API_KEY" in captured.err
