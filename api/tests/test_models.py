import pytest
from pydantic import ValidationError

from newsninja.models import Article, ArticleAnalysis, Briefing, Claim, Entity


def test_article_requires_title_and_url():
    article = Article(title="Apple rises", url="https://example.com/a", source="google_news")
    assert article.body == ""
    assert article.published is None


def test_entity_kind_is_constrained():
    with pytest.raises(ValidationError):
        Entity(name="Apple", kind="corporation")


def test_claim_carries_a_quote():
    claim = Claim(text="Apple stock rose", quote="Apple stock surged 8%")
    assert claim.quote == "Apple stock surged 8%"


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        ArticleAnalysis(
            topic="apple",
            summary="s",
            entities=[],
            stance="positive",
            confidence=1.5,
            key_claims=[],
        )


def test_stance_is_constrained():
    with pytest.raises(ValidationError):
        ArticleAnalysis(
            topic="apple",
            summary="s",
            entities=[],
            stance="Positive",  # capitalised — must fail
            confidence=0.5,
            key_claims=[],
        )


def test_briefing_defaults_to_english():
    briefing = Briefing(topics=["apple"], script="hello", analyses=[])
    assert briefing.language == "en"
