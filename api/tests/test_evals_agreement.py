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


def test_repeated_entities_are_deduplicated_before_scoring():
    """Four emissions of one name score as one entity, so repetition cannot buy
    precision. ``evals.metrics.mean_entities_per_topic`` counts all four — the
    two families answer different questions and are documented as differing."""
    m = agreement_metrics([_result("ai", ["Apple", "apple", "APPLE", "Apple"], "positive")],
                          [_label("ai", ["Apple"], "positive")])
    assert m.entity_precision == 1.0
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


def test_kappa_uses_each_raters_own_marginals():
    """Expected agreement multiplies rater A's class frequency by rater B's.
    Using either rater's marginals twice is a different statistic."""
    k = cohens_kappa(["a", "a", "a", "b"], ["a", "b", "b", "b"])
    # observed 0.5; expected (3/4)(1/4) + (1/4)(3/4) = 0.375
    assert k == pytest.approx(0.2)


def test_kappa_is_unavailable_for_a_single_item():
    """Kappa is undefined at n=1: there are no marginals to discount. The
    arithmetic would return a confident 0.0 for a sample of one."""
    assert cohens_kappa(["a"], ["b"]) is None
    assert cohens_kappa(["a"], ["a"]) is None


def test_a_single_paired_topic_reports_no_kappa():
    m = agreement_metrics([_result("ai", ["A"], "positive")],
                          [_label("ai", ["A"], "negative")])
    assert m.stance_accuracy == 0.0
    assert m.stance_kappa is None


def test_duplicate_golden_topics_are_rejected():
    """Keeping the last label silently would still count both in
    ``total_labels``, understating how much of the set backs the numbers."""
    with pytest.raises(ValueError, match="duplicate topics in the golden set"):
        agreement_metrics([_result("ai", ["A"], "positive")],
                          [_label("ai", ["A"], "positive"),
                           _label("ai", ["B"], "negative")])


def test_duplicate_result_topics_are_rejected():
    """Two results for one topic would both pair against the same label and
    double-count labelled coverage."""
    with pytest.raises(ValueError, match="duplicate topics in the extraction results"):
        agreement_metrics([_result("ai", ["A"], "positive"),
                           _result("ai", ["B"], "negative")],
                          [_label("ai", ["A"], "positive")])


def test_entity_f1_is_the_harmonic_mean_not_the_arithmetic_one():
    """Precision 0.5 and recall 1.0: harmonic 0.667, arithmetic 0.75. Only the
    harmonic mean refuses to let a high recall paper over a low precision."""
    m = agreement_metrics([_result("ai", ["A", "B", "C", "D"], "positive")],
                          [_label("ai", ["A", "B"], "positive")])
    assert m.entity_precision == pytest.approx(0.5)
    assert m.entity_recall == pytest.approx(1.0)
    assert m.entity_f1 == pytest.approx(2 / 3)


def test_stance_accuracy_divides_by_the_paired_topics_only():
    """Not by every result and not by every label: a failed extraction, an
    unlabelled topic and an unreviewed label are all outside the denominator."""
    results = [
        _result("a", [], "positive"),
        _result("b", [], "positive"),
        ExtractionResult(topic="c", analysis=None, articles=[], failed=True),
        _result("d", [], "positive"),
    ]
    labels = [
        _label("a", [], "positive"),
        _label("b", [], "negative"),
        _label("c", [], "positive"),
        _label("d", [], "positive", reviewed=False),
        _label("z", [], "positive"),
    ]
    m = agreement_metrics(results, labels)
    assert m.labelled_coverage == 2
    assert m.stance_accuracy == 0.5, "1 of 2 paired topics agreed"
