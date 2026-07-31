import pytest
from pydantic import ValidationError

from evals.bootstrap import bootstrap, merge_golden
from evals.corpus import CorpusRecord
from evals.golden import GoldenLabel, load_golden, reviewed_only, save_golden
from newsninja.models import Article, ArticleAnalysis, Claim, Entity


def _label(topic="ai", reviewed=False, entities=None):
    return GoldenLabel(
        topic=topic, entities=entities if entities is not None else ["OpenAI"],
        stance="positive", supported_claim_quotes=["a quote"], reviewed=reviewed,
        provenance="human-corrected" if reviewed else "model-drafted",
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


def test_bootstrap_skips_records_with_no_articles():
    empty = CorpusRecord(topic="empty", articles=[])
    record = CorpusRecord(topic="ai", articles=[
        Article(title="t", url="u", source="google_news", body="models improved")])
    labels = bootstrap(StubClient(), [empty, record])
    assert [l.topic for l in labels] == ["ai"], (
        "a topic with no source articles has no evidence to draft a label from"
    )


def test_reviewed_true_rejects_model_drafted_provenance():
    with pytest.raises(ValidationError):
        GoldenLabel(
            topic="ai", entities=["OpenAI"], stance="positive",
            supported_claim_quotes=["q"], reviewed=True, provenance="model-drafted",
        )


def test_merge_keeps_existing_reviewed_label_untouched():
    reviewed = _label("ai", reviewed=True, entities=["Human-Verified-Entity"])
    fresh_draft = GoldenLabel(
        topic="ai", entities=["Some-New-Entity"], stance="negative",
        supported_claim_quotes=["different quote"], reviewed=False,
        provenance="model-drafted",
    )
    merged = merge_golden([reviewed], [fresh_draft])
    assert len(merged) == 1
    assert merged[0] == reviewed, "a re-run must not overwrite a reviewed label"


def test_merge_replaces_an_unreviewed_draft():
    stale_draft = _label("ai", reviewed=False, entities=["Stale"])
    fresh_draft = GoldenLabel(
        topic="ai", entities=["Fresh"], stance="negative",
        supported_claim_quotes=["different quote"], reviewed=False,
        provenance="model-drafted",
    )
    merged = merge_golden([stale_draft], [fresh_draft])
    assert len(merged) == 1
    assert merged[0].entities == ["Fresh"]


def test_merge_adds_a_new_topic():
    existing = [_label("ai", reviewed=True)]
    fresh_draft = GoldenLabel(
        topic="climate", entities=["Some-Org"], stance="neutral",
        supported_claim_quotes=[], reviewed=False, provenance="model-drafted",
    )
    merged = merge_golden(existing, [fresh_draft])
    assert {l.topic for l in merged} == {"ai", "climate"}
