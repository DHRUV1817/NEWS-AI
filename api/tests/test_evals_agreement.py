import pytest

from evals.agreement import agreement_metrics, cohens_kappa
from evals.golden import GoldenLabel
from evals.metrics import ExtractionResult
from newsninja.models import Article, ArticleAnalysis, Entity


def _result(topic, entities, stance):
    return ExtractionResult(
        topic=topic,
        analysis=ArticleAnalysis(
            topic=topic, summary="s",
            entities=[Entity(name=e, kind="org") for e in entities],
            stance=stance, confidence=0.5, key_claims=[],
        ),
        articles=[Article(title="t", url="u", source="google_news", body="b")],
        failed=False,
    )


def _label(topic, entities, stance, reviewed=True):
    return GoldenLabel(topic=topic, entities=entities, stance=stance,
                       supported_claim_quotes=[], reviewed=reviewed,
                       provenance="human-corrected")


def test_perfect_entity_match_scores_one():
    m = agreement_metrics([_result("ai", ["A", "B"], "positive")],
                          [_label("ai", ["A", "B"], "positive")])
    assert m.entity_precision == 1.0
    assert m.entity_recall == 1.0
    assert m.entity_f1 == 1.0


def test_entity_precision_and_recall_differ_when_prediction_over_generates():
    m = agreement_metrics([_result("ai", ["A", "B", "C"], "positive")],
                          [_label("ai", ["A", "B"], "positive")])
    assert m.entity_precision == pytest.approx(2 / 3)
    assert m.entity_recall == 1.0


def test_total_entity_disagreement_scores_f1_zero_not_unavailable():
    """Precision 0 and recall 0 are measured results. Reporting the worst F1
    as "unavailable" would hide the one number a reader most needs."""
    m = agreement_metrics([_result("ai", ["Wrong"], "positive")],
                          [_label("ai", ["Right"], "positive")])
    assert m.entity_precision == 0.0
    assert m.entity_recall == 0.0
    assert m.entity_f1 == 0.0


def test_entity_matching_is_case_insensitive():
    m = agreement_metrics([_result("ai", ["openai"], "positive")],
                          [_label("ai", ["OpenAI"], "positive")])
    assert m.entity_recall == 1.0


def test_unreviewed_labels_are_ignored_entirely():
    m = agreement_metrics([_result("ai", ["A"], "positive")],
                          [_label("ai", ["A"], "positive", reviewed=False)])
    assert m.entity_f1 is None
    assert m.unavailable_reason is not None
    assert m.labelled_coverage == 0


def test_no_labels_reports_unavailable_not_zero():
    m = agreement_metrics([_result("ai", ["A"], "positive")], [])
    assert m.stance_accuracy is None
    assert m.entity_f1 is None
    assert "no reviewed" in m.unavailable_reason.lower()


def test_stance_accuracy_counts_exact_matches():
    results = [_result("a", [], "positive"), _result("b", [], "negative")]
    labels = [_label("a", [], "positive"), _label("b", [], "neutral")]
    m = agreement_metrics(results, labels)
    assert m.stance_accuracy == 0.5


def test_kappa_is_zero_for_chance_agreement():
    # Both raters assign the same single class to everything: no information.
    assert cohens_kappa(["a", "a", "a"], ["a", "a", "a"]) is None


def test_kappa_is_one_for_perfect_agreement_across_classes():
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)


def test_kappa_is_negative_for_systematic_disagreement():
    k = cohens_kappa(["a", "b"], ["b", "a"])
    assert k is not None and k < 0
