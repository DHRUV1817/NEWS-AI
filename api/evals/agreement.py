"""Metrics measured against human-reviewed labels.

Stance is subjective, so it is reported with Cohen's kappa alongside raw
accuracy — kappa discounts the agreement you would get by chance, which raw
accuracy silently credits.

Kappa is hand-rolled rather than imported so the harness needs no scientific
stack for fifteen lines of arithmetic.

Entities here are compared as case-folded *sets*: a topic whose extraction
names "Apple" four times contributes one entity to precision and recall, so a
model that repeats itself cannot buy agreement by volume. That is deliberately
different from ``evals.metrics.mean_entities_per_topic``, which counts every
emission — the two families answer different questions and their entity counts
will not match on the same data. See the note on ``DeterministicMetrics``.
"""

from collections import Counter
from dataclasses import dataclass

from evals.golden import GoldenLabel, reviewed_only
from evals.metrics import ExtractionResult


@dataclass
class AgreementMetrics:
    labelled_coverage: int
    total_labels: int
    entity_precision: float | None
    entity_recall: float | None
    entity_f1: float | None
    stance_accuracy: float | None
    stance_kappa: float | None
    unavailable_reason: str | None


def cohens_kappa(rater_a: list[str], rater_b: list[str]) -> float | None:
    """Cohen's kappa. ``None`` when it is undefined.

    Undefined happens two ways. Expected agreement of 1.0 — every item in one
    class — divides by zero. And fewer than two items gives the marginals
    nothing to describe: at n=1 with disagreement the arithmetic yields a tidy
    0.0 that reads as "chance-level agreement, measured" when nothing was
    measured at all. Both are a real "cannot say", not a zero.
    """
    if len(rater_a) < 2 or len(rater_a) != len(rater_b):
        return None

    n = len(rater_a)
    observed = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n

    count_a = Counter(rater_a)
    count_b = Counter(rater_b)
    expected = sum(
        (count_a[label] / n) * (count_b[label] / n)
        for label in set(count_a) | set(count_b)
    )

    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def _duplicate_topics(topics: list[str]) -> list[str]:
    return sorted(topic for topic, count in Counter(topics).items() if count > 1)


def agreement_metrics(
    results: list[ExtractionResult], labels: list[GoldenLabel]
) -> AgreementMetrics:
    """Score predictions against human-reviewed labels only.

    Duplicate topics are rejected rather than tolerated. A golden set with two
    labels for one topic would silently keep the last and still count both in
    ``total_labels``, understating how much of the set actually backs the
    numbers; two results for one topic would both pair against the same label
    and double-count ``labelled_coverage``. Either way the reported denominator
    stops meaning what the report says it means, and quietly picking a winner
    is a worse answer than saying the input is malformed.
    """
    duplicate_labels = _duplicate_topics([label.topic for label in labels])
    if duplicate_labels:
        raise ValueError(
            "duplicate topics in the golden set: "
            f"{', '.join(duplicate_labels)}; each topic needs exactly one label"
        )

    duplicate_results = _duplicate_topics([result.topic for result in results])
    if duplicate_results:
        raise ValueError(
            "duplicate topics in the extraction results: "
            f"{', '.join(duplicate_results)}; each topic is evaluated once"
        )

    reviewed = reviewed_only(labels)
    by_topic = {label.topic: label for label in reviewed}

    paired = [
        (r, by_topic[r.topic])
        for r in results
        if not r.failed and r.analysis is not None and r.topic in by_topic
    ]

    if not paired:
        return AgreementMetrics(
            labelled_coverage=0,
            total_labels=len(labels),
            entity_precision=None,
            entity_recall=None,
            entity_f1=None,
            stance_accuracy=None,
            stance_kappa=None,
            unavailable_reason=(
                f"no reviewed labels matched the evaluated topics "
                f"({len(reviewed)}/{len(labels)} labels reviewed)"
            ),
        )

    true_positives = 0
    predicted_total = 0
    actual_total = 0
    predicted_stances: list[str] = []
    actual_stances: list[str] = []

    for result, label in paired:
        analysis = result.analysis
        assert analysis is not None  # narrowed above
        predicted = {e.name.casefold() for e in analysis.entities}
        actual = {name.casefold() for name in label.entities}
        true_positives += len(predicted & actual)
        predicted_total += len(predicted)
        actual_total += len(actual)
        predicted_stances.append(analysis.stance)
        actual_stances.append(label.stance)

    precision = true_positives / predicted_total if predicted_total else None
    recall = true_positives / actual_total if actual_total else None
    # `is None`, not truthiness: a precision or recall of exactly 0.0 is a
    # measured result, and the worst F1 is the one number a report about
    # honesty must not hide behind "unavailable".
    f1: float | None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    accuracy = sum(
        1 for p, a in zip(predicted_stances, actual_stances, strict=True) if p == a
    ) / len(paired)

    return AgreementMetrics(
        labelled_coverage=len(paired),
        total_labels=len(labels),
        entity_precision=precision,
        entity_recall=recall,
        entity_f1=f1,
        stance_accuracy=accuracy,
        stance_kappa=cohens_kappa(predicted_stances, actual_stances),
        unavailable_reason=None,
    )
