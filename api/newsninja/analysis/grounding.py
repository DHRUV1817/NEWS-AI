"""Quote grounding, measured rather than asserted.

Every ``Claim`` carries a ``quote`` the model was told to copy verbatim from a
source article. That design choice is only worth anything if someone checks it,
so this module does the check: exact substring containment against the
concatenated article bodies. No normalisation, no fuzzy matching, no model in
the loop — a quote the model paraphrased or invented does not appear in the
corpus and is counted as ungrounded.

This is a measurement utility for the eval harness. It is deliberately not
wired into the pipeline's control flow: nothing here rewrites, filters, or
blocks a briefing.
"""

from newsninja.models import Article, ArticleAnalysis, Claim


def source_corpus(articles: list[Article]) -> str:
    """The text a quote must appear in: every article body plus its title."""
    return "\n".join(f"{article.title}\n{article.body}" for article in articles)


def ungrounded_claims(
    analysis: ArticleAnalysis, articles: list[Article]
) -> list[Claim]:
    """Claims whose quote does not appear verbatim in the sources."""
    corpus = source_corpus(articles)
    return [claim for claim in analysis.key_claims if claim.quote not in corpus]


def grounding_rate(analysis: ArticleAnalysis, articles: list[Article]) -> float:
    """Fraction of claims whose quote appears verbatim in the sources, 0.0-1.0.

    An analysis with no claims returns 1.0: there is nothing unsupported in it.
    Read that alongside the claim count rather than on its own — an empty
    analysis is vacuously grounded, not well grounded.
    """
    total = len(analysis.key_claims)
    if total == 0:
        return 1.0
    return (total - len(ungrounded_claims(analysis, articles))) / total
