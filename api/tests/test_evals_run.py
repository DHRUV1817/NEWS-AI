"""The runner decides what counts as an evaluated topic.

Every metric denominator in the report is built from what `extract_results`
returns, so anything it records that no extraction actually produced inflates
the published numbers.
"""

from evals.agreement import agreement_metrics
from evals.corpus import CorpusRecord
from evals.golden import GoldenLabel
from evals.metrics import deterministic_metrics
from evals.run import extract_results
from newsninja.errors import ExtractionFailure
from newsninja.models import Article, ArticleAnalysis, Entity


def _article(body: str = "body text") -> Article:
    return Article(title="t", url="u", source="google_news", body=body)


def _analysis(topic: str, stance: str = "positive", entities=("A",)) -> ArticleAnalysis:
    return ArticleAnalysis(
        topic=topic, summary="s",
        entities=[Entity(name=e, kind="org") for e in entities],
        stance=stance, confidence=0.5, key_claims=[],
    )


def _label(topic: str, stance: str = "neutral", entities=()) -> GoldenLabel:
    return GoldenLabel(topic=topic, entities=list(entities), stance=stance,
                       supported_claim_quotes=[], reviewed=True,
                       provenance="human-corrected")


class StubClient:
    """Canned structured responses keyed by the topic in the rendered prompt."""

    def __init__(self, by_topic):
        self.by_topic = by_topic
        self.calls: list[str] = []

    def structured(self, *, model, system, user, schema_model, max_retries=2):
        topic = user.splitlines()[0].removeprefix("Topic: ")
        self.calls.append(topic)
        outcome = self.by_topic[topic]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_a_record_with_no_articles_is_never_extracted_or_recorded():
    client = StubClient({"ai": _analysis("ai")})
    results = extract_results(client, [
        CorpusRecord(topic="quiet", articles=[]),
        CorpusRecord(topic="ai", articles=[_article()]),
    ])
    assert [r.topic for r in results] == ["ai"]
    assert client.calls == ["ai"], "the empty topic must not reach the model either"


def test_a_record_with_no_articles_cannot_inflate_schema_validity():
    """`extract_topic` answers an empty article list with a synthetic analysis
    and no model call. Counting it would report a schema-valid extraction that
    never happened."""
    client = StubClient({"ai": ExtractionFailure("no valid output")})
    results = extract_results(client, [
        CorpusRecord(topic="quiet", articles=[]),
        CorpusRecord(topic="ai", articles=[_article()]),
    ])
    m = deterministic_metrics(results)
    assert m.topics_evaluated == 1
    assert m.schema_valid_rate == 0.0, "the only real extraction failed"


def test_a_record_with_no_articles_cannot_inflate_stance_accuracy():
    """The synthetic analysis carries a hard-coded ``stance="neutral"``. Scored
    against a reviewed ``neutral`` label it would earn free agreement."""
    client = StubClient({"ai": _analysis("ai", stance="positive")})
    results = extract_results(client, [
        CorpusRecord(topic="quiet", articles=[]),
        CorpusRecord(topic="ai", articles=[_article()]),
    ])
    m = agreement_metrics(results, [_label("quiet"), _label("ai")])
    assert m.labelled_coverage == 1
    assert m.stance_accuracy == 0.0, "the only real prediction disagreed"


def test_a_failed_extraction_is_still_recorded_as_failed():
    client = StubClient({"ai": ExtractionFailure("boom")})
    results = extract_results(client, [CorpusRecord(topic="ai", articles=[_article()])])
    assert len(results) == 1
    assert results[0].failed is True
    assert results[0].analysis is None
