from evals.bootstrap import bootstrap
from evals.corpus import CorpusRecord
from evals.golden import GoldenLabel, load_golden, reviewed_only, save_golden
from newsninja.models import Article, ArticleAnalysis, Claim, Entity


def _label(topic="ai", reviewed=False):
    return GoldenLabel(
        topic=topic, entities=["OpenAI"], stance="positive",
        supported_claim_quotes=["a quote"], reviewed=reviewed,
        provenance="model-drafted",
    )


def test_labels_default_to_unreviewed():
    assert GoldenLabel(topic="ai", entities=[], stance="neutral",
                       supported_claim_quotes=[]).reviewed is False


def test_round_trips(tmp_path):
    path = tmp_path / "golden.jsonl"
    save_golden([_label("ai"), _label("climate")], path)
    assert [l.topic for l in load_golden(path)] == ["ai", "climate"]


def test_reviewed_only_filters_out_drafts():
    labels = [_label("ai", reviewed=True), _label("climate", reviewed=False)]
    assert [l.topic for l in reviewed_only(labels)] == ["ai"]


def test_load_missing_returns_empty_rather_than_raising(tmp_path):
    assert load_golden(tmp_path / "absent.jsonl") == []


class StubClient:
    def structured(self, *, model, system, user, schema_model, max_retries=2):
        return ArticleAnalysis(
            topic="ai", summary="s",
            entities=[Entity(name="OpenAI", kind="org")],
            stance="positive", confidence=0.9,
            key_claims=[Claim(text="c", quote="models improved")],
        )


def test_bootstrap_marks_every_draft_unreviewed():
    record = CorpusRecord(topic="ai", articles=[
        Article(title="t", url="u", source="google_news", body="models improved")])
    labels = bootstrap(StubClient(), [record])
    assert len(labels) == 1
    assert labels[0].reviewed is False
    assert labels[0].provenance == "model-drafted"
    assert labels[0].entities == ["OpenAI"]


def test_bootstrap_only_keeps_quotes_that_are_actually_in_the_source():
    record = CorpusRecord(topic="ai", articles=[
        Article(title="t", url="u", source="google_news", body="something else entirely")])
    labels = bootstrap(StubClient(), [record])
    assert labels[0].supported_claim_quotes == [], (
        "a drafted quote absent from the source must not become ground truth"
    )
