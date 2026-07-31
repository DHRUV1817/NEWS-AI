import pytest

from newsninja.analysis.grounding import grounding_rate, ungrounded_claims
from newsninja.models import Article, ArticleAnalysis, Claim


def _articles():
    return [
        Article(
            title="Chips",
            url="u1",
            source="google_news",
            body="Nvidia reported record revenue this quarter.",
        ),
        Article(
            title="Policy",
            url="u2",
            source="reddit",
            body="The regulator opened a formal inquiry on Tuesday.",
        ),
    ]


def _analysis(*quotes: str) -> ArticleAnalysis:
    return ArticleAnalysis(
        topic="ai",
        summary="s",
        entities=[],
        stance="neutral",
        confidence=0.5,
        key_claims=[Claim(text=f"claim {i}", quote=q) for i, q in enumerate(quotes)],
    )


def test_every_quote_present_scores_one():
    analysis = _analysis(
        "Nvidia reported record revenue", "opened a formal inquiry on Tuesday"
    )
    assert grounding_rate(analysis, _articles()) == 1.0


def test_an_invented_quote_scores_zero():
    analysis = _analysis("Nvidia announced a merger with Intel")
    assert grounding_rate(analysis, _articles()) == 0.0


def test_partial_grounding_is_a_fraction():
    analysis = _analysis(
        "Nvidia reported record revenue",  # verbatim
        "the regulator opened an inquiry",  # paraphrased: different wording
    )
    assert grounding_rate(analysis, _articles()) == pytest.approx(0.5)


def test_containment_is_exact_not_normalised():
    """Whitespace and case changes are still not the source text."""
    analysis = _analysis("nvidia reported record revenue")
    assert grounding_rate(analysis, _articles()) == 0.0


def test_a_quote_may_span_the_title():
    articles = [Article(title="Chips rally", url="u", source="s", body="Prices rose.")]
    assert grounding_rate(_analysis("Chips rally"), articles) == 1.0


def test_no_claims_is_vacuously_grounded():
    assert grounding_rate(_analysis(), _articles()) == 1.0


def test_no_articles_grounds_nothing():
    assert grounding_rate(_analysis("anything at all"), []) == 0.0


def test_ungrounded_claims_names_the_offenders():
    analysis = _analysis("Nvidia reported record revenue", "a fabricated line")
    offenders = ungrounded_claims(analysis, _articles())
    assert [claim.quote for claim in offenders] == ["a fabricated line"]
