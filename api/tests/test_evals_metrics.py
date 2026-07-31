from evals.metrics import ExtractionResult, deterministic_metrics
from newsninja.analysis.grounding import grounding_rate as per_topic_grounding_rate
from newsninja.models import Article, ArticleAnalysis, Claim, Entity


def _article(body: str) -> Article:
    return Article(title="t", url="u", source="google_news", body=body)


def _analysis(quotes: list[str], entities: int = 1) -> ArticleAnalysis:
    return ArticleAnalysis(
        topic="ai",
        summary="s",
        entities=[Entity(name=f"E{i}", kind="org") for i in range(entities)],
        stance="neutral",
        confidence=0.5,
        key_claims=[Claim(text=f"c{i}", quote=q) for i, q in enumerate(quotes)],
    )


def test_all_quotes_grounded_gives_rate_one():
    result = ExtractionResult(
        topic="ai", analysis=_analysis(["alpha", "beta"]),
        articles=[_article("alpha and beta appear here")], failed=False,
    )
    m = deterministic_metrics([result])
    assert m.grounding_rate == 1.0
    assert m.ungrounded_claim_rate == 0.0


def test_a_fabricated_quote_is_counted_as_ungrounded():
    result = ExtractionResult(
        topic="ai", analysis=_analysis(["alpha", "never said this"]),
        articles=[_article("alpha appears here")], failed=False,
    )
    m = deterministic_metrics([result])
    assert m.grounding_rate == 0.5
    assert m.ungrounded_claim_rate == 0.5


def test_schema_valid_rate_counts_failed_extractions():
    ok = ExtractionResult(topic="a", analysis=_analysis(["x"]),
                          articles=[_article("x")], failed=False)
    bad = ExtractionResult(topic="b", analysis=None, articles=[_article("y")], failed=True)
    m = deterministic_metrics([ok, bad])
    assert m.schema_valid_rate == 0.5
    assert m.topics_evaluated == 2


def test_grounding_ignores_failed_extractions_rather_than_scoring_them_zero():
    ok = ExtractionResult(topic="a", analysis=_analysis(["x"]),
                          articles=[_article("x")], failed=False)
    bad = ExtractionResult(topic="b", analysis=None, articles=[_article("y")], failed=True)
    m = deterministic_metrics([ok, bad])
    assert m.grounding_rate == 1.0, "a failed extraction is not an ungrounded one"


def test_empty_input_reports_unavailable_not_zero():
    m = deterministic_metrics([])
    assert m.grounding_rate is None
    assert m.schema_valid_rate is None
    assert m.topics_evaluated == 0


def test_mean_counts_are_reported():
    result = ExtractionResult(topic="ai", analysis=_analysis(["a", "b"], entities=3),
                              articles=[_article("a b")], failed=False)
    m = deterministic_metrics([result])
    assert m.mean_claims_per_topic == 2.0
    assert m.mean_entities_per_topic == 3.0


def test_aggregate_grounding_matches_per_topic_formula_exactly():
    """The aggregate must use the same arithmetic shape as the per-topic
    ``newsninja.analysis.grounding.grounding_rate`` so a report citing both
    numbers for identical data never prints two different rates from
    algebraically-equivalent but not bit-identical formulas."""
    analysis = _analysis(["alpha", "nope", "beta"])
    articles = [_article("alpha beta appear here")]
    result = ExtractionResult(topic="ai", analysis=analysis, articles=articles, failed=False)
    m = deterministic_metrics([result])
    assert m.grounding_rate == per_topic_grounding_rate(analysis, articles)


def test_zero_claims_reports_unavailable_not_perfect():
    """An extraction that succeeded but produced no claims must not be read
    as 100% grounded — there is no evidence to be grounded or ungrounded."""
    result = ExtractionResult(
        topic="ai", analysis=_analysis([], entities=2),
        articles=[_article("x")], failed=False,
    )
    m = deterministic_metrics([result])
    assert m.grounding_rate is None
    assert m.ungrounded_claim_rate is None


def test_all_failed_results_report_unavailable_without_crashing():
    bad_a = ExtractionResult(topic="a", analysis=None, articles=[_article("x")], failed=True)
    bad_b = ExtractionResult(topic="b", analysis=None, articles=[_article("y")], failed=True)
    m = deterministic_metrics([bad_a, bad_b])
    assert m.schema_valid_rate == 0.0
    assert m.grounding_rate is None
    assert m.ungrounded_claim_rate is None
    assert m.mean_claims_per_topic is None
    assert m.mean_entities_per_topic is None
