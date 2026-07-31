"""Topic extraction.

All articles for one topic go into a single structured call. At eight articles
per topic and five topics, per-article calls would be forty requests and roughly
32,000 tokens — well past the 8,000 TPM free-tier ceiling. Batching makes it five.
"""

from newsninja.analysis.client import StructuredClient
from newsninja.analysis.prompts import EXTRACT_SYSTEM, PROMPT_VERSION
from newsninja.cache import Cache
from newsninja.models import Article, ArticleAnalysis

DEFAULT_MODEL = "openai/gpt-oss-20b"


def _render(topic: str, articles: list[Article]) -> str:
    lines = [f"Topic: {topic}", "", "Articles:"]
    for index, article in enumerate(articles, start=1):
        lines.append(f"[{index}] ({article.source}) {article.title}")
        lines.append(article.body)
        lines.append("")
    return "\n".join(lines)


def _empty(topic: str) -> ArticleAnalysis:
    return ArticleAnalysis(
        topic=topic,
        summary=f"No articles were found for {topic}.",
        entities=[],
        stance="neutral",
        confidence=0.0,
        key_claims=[],
    )


def extract_topic(
    client: StructuredClient,
    topic: str,
    articles: list[Article],
    cache: Cache | None = None,
    model: str = DEFAULT_MODEL,
) -> ArticleAnalysis:
    """Consolidate ``articles`` into one typed analysis of ``topic``."""
    if not articles:
        return _empty(topic)

    user = _render(topic, articles)

    key = None
    if cache is not None:
        key = Cache.make_key(
            kind="extract",
            topic=topic,
            model=model,
            prompt_version=PROMPT_VERSION,
            payload=user,
        )
        cached = cache.get(key)
        if cached is not None:
            return ArticleAnalysis.model_validate_json(cached)

    analysis = client.structured(
        model=model,
        system=EXTRACT_SYSTEM,
        user=user,
        schema_model=ArticleAnalysis,
    )

    if cache is not None and key is not None:
        cache.set(key, analysis.model_dump_json())

    return analysis
