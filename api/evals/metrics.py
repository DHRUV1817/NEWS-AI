"""Metrics that need no ground truth.

Everything here is arithmetic over the model's own output and its source
articles. No model judges any of it, which is what makes these the numbers
the README can cite without qualification.

A metric that cannot be computed is ``None``, never ``0.0`` — reporting a
zero for "no data" would understate quality as confidently as inventing a
number would overstate it.
"""

from dataclasses import dataclass

from pydantic import BaseModel

from newsninja.analysis.grounding import ungrounded_claims
from newsninja.models import Article, ArticleAnalysis


class ExtractionResult(BaseModel):
    """One topic's extraction outcome, successful or not."""

    topic: str
    analysis: ArticleAnalysis | None
    articles: list[Article]
    failed: bool


@dataclass
class DeterministicMetrics:
    topics_evaluated: int
    schema_valid_rate: float | None
    grounding_rate: float | None
    hallucinated_claim_rate: float | None
    mean_claims_per_topic: float | None
    mean_entities_per_topic: float | None


def deterministic_metrics(results: list[ExtractionResult]) -> DeterministicMetrics:
    """Compute label-free metrics over extraction results."""
    if not results:
        return DeterministicMetrics(0, None, None, None, None, None)

    succeeded = [r for r in results if not r.failed and r.analysis is not None]
    schema_valid_rate = len(succeeded) / len(results)

    if not succeeded:
        return DeterministicMetrics(
            len(results), schema_valid_rate, None, None, None, None
        )

    total_claims = 0
    total_ungrounded = 0
    total_entities = 0
    for result in succeeded:
        analysis = result.analysis
        assert analysis is not None  # narrowed by the filter above
        total_claims += len(analysis.key_claims)
        total_ungrounded += len(ungrounded_claims(analysis, result.articles))
        total_entities += len(analysis.entities)

    grounding = None if total_claims == 0 else 1.0 - (total_ungrounded / total_claims)
    hallucinated = None if total_claims == 0 else total_ungrounded / total_claims

    return DeterministicMetrics(
        topics_evaluated=len(results),
        schema_valid_rate=schema_valid_rate,
        grounding_rate=grounding,
        hallucinated_claim_rate=hallucinated,
        mean_claims_per_topic=total_claims / len(succeeded),
        mean_entities_per_topic=total_entities / len(succeeded),
    )
