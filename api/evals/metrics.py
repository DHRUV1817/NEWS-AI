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
    """Label-free metrics aggregated across extraction results.

    ``ungrounded_claim_rate`` measures quote-absence only: a claim whose
    ``text`` is fabricated but whose ``quote`` happens to be a real substring
    of the corpus still counts as grounded here. It is not a general
    hallucination detector — it is precisely ``1 - grounding_rate``.

    ``mean_entities_per_topic`` counts every entity the model emitted,
    duplicates included: a topic naming "Apple" four times contributes four.
    It describes output volume, not distinct coverage. ``evals.agreement``
    deliberately does the opposite — it case-folds entities into a set before
    scoring precision and recall, so those same four emissions count once.
    The two families therefore report different entity counts for identical
    data, on purpose: one asks how much the model said, the other how much of
    what it said was right.

    ``total_claims`` and ``total_ungrounded_claims`` are the denominator and
    numerator ``grounding_rate`` and ``ungrounded_claim_rate`` were divided
    from. A rate is a fraction with the count discarded; a rate of 1.00 over
    three claims and a rate of 1.00 over three thousand claims are not the
    same finding, and only the count tells them apart. Unlike the rates,
    these are never ``None`` — a count of zero claims is a true fact about
    the run, not a missing measurement, so it is reported as ``0``.
    """

    topics_evaluated: int
    schema_valid_rate: float | None
    grounding_rate: float | None
    ungrounded_claim_rate: float | None
    total_claims: int
    total_ungrounded_claims: int
    mean_claims_per_topic: float | None
    mean_entities_per_topic: float | None


def deterministic_metrics(results: list[ExtractionResult]) -> DeterministicMetrics:
    """Compute label-free metrics over extraction results."""
    if not results:
        return DeterministicMetrics(0, None, None, None, 0, 0, None, None)

    succeeded = [r for r in results if not r.failed and r.analysis is not None]
    schema_valid_rate = len(succeeded) / len(results)

    if not succeeded:
        return DeterministicMetrics(
            len(results), schema_valid_rate, None, None, 0, 0, None, None
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

    # Use the same (total - ungrounded) / total shape as
    # newsninja.analysis.grounding.grounding_rate so a per-topic rate and this
    # aggregate agree exactly on identical data, rather than differing in the
    # last bit from an algebraically-equivalent but not bit-identical formula.
    grounded = None if total_claims == 0 else (total_claims - total_ungrounded) / total_claims
    ungrounded = None if total_claims == 0 else total_ungrounded / total_claims

    return DeterministicMetrics(
        topics_evaluated=len(results),
        schema_valid_rate=schema_valid_rate,
        grounding_rate=grounded,
        ungrounded_claim_rate=ungrounded,
        total_claims=total_claims,
        total_ungrounded_claims=total_ungrounded,
        mean_claims_per_topic=total_claims / len(succeeded),
        mean_entities_per_topic=total_entities / len(succeeded),
    )
