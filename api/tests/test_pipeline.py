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
        if not self._available:
            # The pipeline must never reach here. Raising rather than returning
            # nothing is what makes the availability guard's removal detectable.
            raise AssertionError(f"{self.name} is unavailable and must not be fetched")
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


def test_every_failure_of_a_source_is_kept_not_just_the_last():
    """Regression: source_errors[name] was overwritten once per topic."""

    class FlakySource:
        name = "reddit"

        def __init__(self):
            self.calls = 0

        def available(self):
            return True

        def fetch(self, topic, limit=8):
            self.calls += 1
            raise SourceError(self.name, f"failure {self.calls}")

    result = run_pipeline(
        topics=["ai", "climate"],
        sources=[FlakySource(), FakeSource("google_news", [_article()])],
        client=FakeClient(),
        tts=_tts,
    )
    assert result.source_errors["reddit"] == [
        "reddit: failure 1",
        "reddit: failure 2",
    ]


def test_unavailable_sources_are_skipped_without_error():
    """The unavailable source has articles to give and explodes if fetched, so
    removing the availability guard in ``run_pipeline`` fails this test."""
    result = run_pipeline(
        topics=["ai"],
        sources=[
            FakeSource("reddit", [_article("reddit")], is_available=False),
            FakeSource("google_news", [_article()]),
        ],
        client=FakeClient(),
        tts=_tts,
    )
    assert result.source_errors == {}
    assert result.skipped_sources == ["reddit"]


def test_a_skipped_source_is_listed_once_per_run_not_once_per_topic():
    result = run_pipeline(
        topics=["ai", "climate"],
        sources=[
            FakeSource("reddit", [_article("reddit")], is_available=False),
            FakeSource("google_news", [_article()]),
        ],
        client=FakeClient(),
        tts=_tts,
    )
    assert result.skipped_sources == ["reddit"]


def test_available_sources_are_not_reported_as_skipped():
    result = run_pipeline(
        topics=["ai"],
        sources=[FakeSource("google_news", [_article()])],
        client=FakeClient(),
        tts=_tts,
    )
    assert result.skipped_sources == []


def test_default_audio_path_wires_orpheus_when_enabled_and_keyed(monkeypatch):
    """Regression: `enable_orpheus=True` used to yield gTTS silently.

    ``run_pipeline``'s own speech seam is exercised — no ``tts`` argument — so
    this fails if the default path never passes an ``orpheus_fn``.
    """
    import newsninja.pipeline as pipeline_module

    captured: dict = {}
    sentinel = object()

    def fake_make_orpheus_fn(api_key, post=None):
        captured["api_key"] = api_key
        return sentinel

    def fake_synthesize_speech(text, language, enable_orpheus, orpheus_fn=None):
        captured["orpheus_fn"] = orpheus_fn
        return b"AUDIO"

    monkeypatch.setattr(pipeline_module, "make_orpheus_fn", fake_make_orpheus_fn)
    monkeypatch.setattr(pipeline_module, "synthesize_speech", fake_synthesize_speech)

    run_pipeline(
        topics=["ai"],
        sources=[FakeSource("google_news", [_article()])],
        client=FakeClient(),
        enable_orpheus=True,
        api_key="sk-test",
    )

    assert captured["api_key"] == "sk-test"
    assert captured["orpheus_fn"] is sentinel


def test_default_audio_path_skips_orpheus_without_a_key(monkeypatch):
    import newsninja.pipeline as pipeline_module

    captured: dict = {}

    def fake_synthesize_speech(text, language, enable_orpheus, orpheus_fn=None):
        captured["orpheus_fn"] = orpheus_fn
        return b"AUDIO"

    monkeypatch.setattr(pipeline_module, "synthesize_speech", fake_synthesize_speech)

    run_pipeline(
        topics=["ai"],
        sources=[FakeSource("google_news", [_article()])],
        client=FakeClient(),
        enable_orpheus=True,
    )

    assert captured["orpheus_fn"] is None


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


def _briefing():
    from newsninja.models import Briefing

    return Briefing(topics=["ai"], script="Here is your briefing.", analyses=[])


def _patch_cli(monkeypatch, run_pipeline_impl):
    """Point the CLI at a stub pipeline and at Settings that ignore ../.env.

    The real .env on a developer machine holds a live key; `cli.main` imports
    both Settings and run_pipeline inside the function body, so patching the
    module attributes takes effect at call time.
    """
    from pydantic_settings import SettingsConfigDict

    import newsninja.config as config_module
    import newsninja.pipeline as pipeline_module

    class NoEnvFileSettings(config_module.Settings):
        model_config = SettingsConfigDict(env_file=None, extra="ignore")

    monkeypatch.setattr(config_module, "Settings", NoEnvFileSettings)
    monkeypatch.setattr(pipeline_module, "run_pipeline", run_pipeline_impl)


def test_cli_reports_skipped_sources_distinctly_from_failures(capsys, monkeypatch):
    """With Reddit credentials absent — the normal state — say so."""
    from newsninja.cli import main
    from newsninja.pipeline import PipelineResult

    def fake_run_pipeline(**kwargs):
        return PipelineResult(
            briefing=_briefing(),
            audio=b"",
            source_errors={"google_news": ["google_news: 503"]},
            skipped_sources=["reddit"],
        )

    _patch_cli(monkeypatch, fake_run_pipeline)

    exit_code = main(["--topic", "ai", "--no-audio", "--no-cache"])
    err = capsys.readouterr().err

    assert exit_code == 0
    assert "reddit" in err
    assert "skipped" in err
    assert "warning: source google_news failed" in err
    assert "reddit was skipped" in err.replace("source ", "")


def test_cli_reports_every_failure_of_a_repeatedly_failing_source(capsys, monkeypatch):
    from newsninja.cli import main
    from newsninja.pipeline import PipelineResult

    def fake_run_pipeline(**kwargs):
        return PipelineResult(
            briefing=_briefing(),
            audio=b"",
            source_errors={"reddit": ["reddit: 401", "reddit: 503"]},
        )

    _patch_cli(monkeypatch, fake_run_pipeline)

    main(["--topic", "ai", "--no-audio", "--no-cache"])
    err = capsys.readouterr().err
    assert "401" in err
    assert "503" in err


def test_cli_reports_a_rate_limit_readably_with_a_nonzero_exit(capsys, monkeypatch):
    """Regression: the only shipped interface showed a traceback."""
    from newsninja.cli import main
    from newsninja.errors import RateLimitError

    def fake_run_pipeline(**kwargs):
        raise RateLimitError("limit exceeded for gpt-oss-20b", retry_after=42.0)

    _patch_cli(monkeypatch, fake_run_pipeline)

    exit_code = main(["--topic", "ai", "--no-audio", "--no-cache"])
    err = capsys.readouterr().err

    assert exit_code != 0
    assert "rate limit" in err.lower()
    assert "42" in err, "the user needs to know how long to wait"
    assert "Traceback" not in err


def test_cli_reports_an_extraction_failure_readably(capsys, monkeypatch):
    from newsninja.cli import main
    from newsninja.errors import ExtractionFailure

    def fake_run_pipeline(**kwargs):
        raise ExtractionFailure("ArticleAnalysis did not validate after 3 attempts")

    _patch_cli(monkeypatch, fake_run_pipeline)

    exit_code = main(["--topic", "ai", "--no-audio", "--no-cache"])
    err = capsys.readouterr().err

    assert exit_code != 0
    assert "did not validate" in err
    assert "Traceback" not in err


def test_cli_uses_distinct_exit_codes_for_distinct_failures(monkeypatch):
    from newsninja.cli import EXIT_EXTRACTION, EXIT_RATE_LIMIT, main
    from newsninja.errors import ExtractionFailure, RateLimitError

    def raising(exc):
        def _run(**kwargs):
            raise exc

        return _run

    _patch_cli(monkeypatch, raising(RateLimitError("limited", retry_after=1.0)))
    rate_limited = main(["--topic", "ai", "--no-audio", "--no-cache"])

    _patch_cli(monkeypatch, raising(ExtractionFailure("no")))
    extraction = main(["--topic", "ai", "--no-audio", "--no-cache"])

    assert rate_limited == EXIT_RATE_LIMIT
    assert extraction == EXIT_EXTRACTION
    assert rate_limited != extraction


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
