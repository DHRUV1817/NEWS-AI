from evals.metrics import ExtractionResult, deterministic_metrics
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
    assert m.hallucinated_claim_rate == 0.0


def test_a_fabricated_quote_is_counted_as_hallucinated():
    result = ExtractionResult(
        topic="ai", analysis=_analysis(["alpha", "never said this"]),
        articles=[_article("alpha appears here")], failed=False,
    )
    m = deterministic_metrics([result])
    assert m.grounding_rate == 0.5
    assert m.hallucinated_claim_rate == 0.5


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
