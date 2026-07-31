import pytest

from evals.judge import JUDGE_MODEL, JudgeScore, judge_summary, judged_metrics
from evals.metrics import ExtractionResult
from newsninja.models import Article, ArticleAnalysis


def _result(topic="ai", summary="A summary."):
    return ExtractionResult(
        topic=topic,
        analysis=ArticleAnalysis(topic=topic, summary=summary, entities=[],
                                 stance="neutral", confidence=0.5, key_claims=[]),
        articles=[Article(title="t", url="u", source="google_news", body="body")],
        failed=False,
    )


class RecordingJudge:
    def __init__(self, score=None):
        self.score = score or {"coverage": 4, "neutrality": 5,
                               "coherence": 4, "rationale": "fine"}
        self.calls: list[dict] = []

    def structured(self, *, model, system, user, schema_model, max_retries=2):
        self.calls.append({"model": model, "user": user})
        return schema_model.model_validate(self.score)


def test_judge_returns_a_score():
    score = judge_summary(RecordingJudge(), _result())
    assert isinstance(score, JudgeScore)
    assert score.coverage == 4


def test_judge_uses_the_reasoning_model():
    client = RecordingJudge()
    judge_summary(client, _result())
    assert client.calls[0]["model"] == JUDGE_MODEL


def test_judge_is_shown_the_sources_not_just_the_summary():
    client = RecordingJudge()
    judge_summary(client, _result())
    assert "body" in client.calls[0]["user"], (
        "a judge that cannot see the sources cannot score coverage"
    )


def test_scores_are_bounded():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        JudgeScore(coverage=6, neutrality=3, coherence=3, rationale="r")


def test_judged_metrics_average_across_results():
    client = RecordingJudge()
    m = judged_metrics(client, [_result("a"), _result("b")])
    assert m.judged_count == 2
    assert m.mean_coverage == 4.0


def test_failed_extractions_are_not_judged():
    client = RecordingJudge()
    bad = ExtractionResult(topic="b", analysis=None, articles=[], failed=True)
    m = judged_metrics(client, [_result("a"), bad])
    assert m.judged_count == 1
    assert len(client.calls) == 1


def test_no_judgeable_results_reports_unavailable():
    m = judged_metrics(RecordingJudge(), [])
    assert m.judged_count == 0
    assert m.mean_coverage is None


class ScriptedJudge:
    """Returns the next scored triple on each call, so per-axis means and the
    order of judgement are both observable."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls: list[dict] = []

    def structured(self, *, model, system, user, schema_model, max_retries=2):
        self.calls.append({"model": model, "user": user})
        coverage, neutrality, coherence = self.scores.pop(0)
        return schema_model.model_validate({
            "coverage": coverage, "neutrality": neutrality,
            "coherence": coherence, "rationale": "r",
        })


def test_each_axis_mean_comes_from_its_own_axis():
    """Distinct scores per axis: a mean wired to the neighbouring field cannot
    pass by coincidence."""
    client = ScriptedJudge([(1, 3, 5)])
    m = judged_metrics(client, [_result("a")])
    assert m.mean_coverage == 1.0
    assert m.mean_neutrality == 3.0
    assert m.mean_coherence == 5.0


def test_each_axis_is_averaged_across_every_judged_summary():
    client = ScriptedJudge([(1, 2, 3), (4, 5, 2)])
    m = judged_metrics(client, [_result("a"), _result("b")])
    assert m.judged_count == 2
    assert m.mean_coverage == pytest.approx(2.5)
    assert m.mean_neutrality == pytest.approx(3.5)
    assert m.mean_coherence == pytest.approx(2.5)


def test_means_divide_by_the_number_judged_not_the_number_of_results():
    """A failed extraction is not a zero-scoring summary. Dividing by the
    result count would silently deflate every judged axis."""
    failed = ExtractionResult(topic="b", analysis=None, articles=[], failed=True)
    client = ScriptedJudge([(3, 4, 5)])
    m = judged_metrics(client, [_result("a"), failed, failed])
    assert m.judged_count == 1
    assert m.mean_coverage == 3.0
    assert m.mean_neutrality == 4.0
    assert m.mean_coherence == 5.0
