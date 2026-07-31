"""Metrics measured against human-reviewed labels.

Stance is subjective, so it is reported with Cohen's kappa alongside raw
accuracy — kappa discounts the agreement you would get by chance, which raw
accuracy silently credits.

Kappa is hand-rolled rather than imported so the harness needs no scientific
stack for fifteen lines of arithmetic.
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

    Undefined happens when expected agreement is 1.0 — every item in one class
    — where the statistic divides by zero. That is a real "cannot say", not a
    zero.
    """
    if not rater_a or len(rater_a) != len(rater_b):
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


def agreement_metrics(
    results: list[ExtractionResult], labels: list[GoldenLabel]
) -> AgreementMetrics:
    """Score predictions against human-reviewed labels only."""
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
