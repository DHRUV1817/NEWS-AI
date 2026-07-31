"""Markdown report generation.

The three metric families are kept visually separate because they carry
different epistemic weight: deterministic numbers are arithmetic, agreement
numbers depend on how many labels a human actually reviewed, and judged numbers
are one model's opinion of another's output.
"""

from typing import Any

from evals.agreement import AgreementMetrics
from evals.judge import JudgedMetrics
from evals.metrics import DeterministicMetrics


def _fmt(value: float | None, digits: int = 2) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def render_report(
    deterministic: DeterministicMetrics,
    agreement: AgreementMetrics,
    judged: JudgedMetrics,
    meta: dict[str, Any],
) -> str:
    lines: list[str] = [
        "# Evaluation report",
        "",
        f"Generated: {meta.get('generated', 'unknown')}  ",
        f"Extraction model: `{meta.get('model', 'unknown')}`  ",
        f"Prompt version: `{meta.get('prompt_version', 'unknown')}`  ",
        f"Topics evaluated: {deterministic.topics_evaluated}",
        "",
        "## Deterministic",
        "",
        (
            "No model judges these. They are arithmetic over the extraction "
            "output and its source articles."
        ),
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Schema validity rate | {_fmt(deterministic.schema_valid_rate)} |",
        f"| Quote grounding rate | {_fmt(deterministic.grounding_rate)} |",
        f"| Ungrounded claim rate | {_fmt(deterministic.ungrounded_claim_rate)} |",
        f"| Mean claims per topic | {_fmt(deterministic.mean_claims_per_topic, 1)} |",
        f"| Mean entities per topic | {_fmt(deterministic.mean_entities_per_topic, 1)} |",
        "",
        "## Agreement",
        "",
        (
            f"Backed by {agreement.labelled_coverage} human-reviewed labels "
            f"out of {agreement.total_labels} in the golden set."
        ),
        "",
    ]

    if agreement.unavailable_reason:
        lines += [
            f"**unavailable** — {agreement.unavailable_reason}.",
            "",
            (
                "Correct drafted labels in `evals/data/golden.jsonl` and set "
                "`reviewed: true` to populate this section."
            ),
            "",
        ]
    else:
        lines += [
            "| Metric | Value |",
            "| --- | --- |",
            f"| Entity precision | {_fmt(agreement.entity_precision)} |",
            f"| Entity recall | {_fmt(agreement.entity_recall)} |",
            f"| Entity F1 | {_fmt(agreement.entity_f1)} |",
            f"| Stance accuracy | {_fmt(agreement.stance_accuracy)} |",
            f"| Stance Cohen's kappa | {_fmt(agreement.stance_kappa)} |",
            "",
            "Kappa discounts chance agreement, which raw accuracy credits.",
            "",
        ]

    lines += [
        "## Judged",
        "",
        (
            "One model's rubric score of another model's output, on a 1-5 "
            "scale. Reported separately because it is not ground truth."
        ),
        "",
        f"Summaries judged: {judged.judged_count}",
        "",
        "| Axis | Mean |",
        "| --- | --- |",
        f"| Coverage | {_fmt(judged.mean_coverage, 1)} |",
        f"| Neutrality | {_fmt(judged.mean_neutrality, 1)} |",
        f"| Coherence | {_fmt(judged.mean_coherence, 1)} |",
        "",
    ]

    return "\n".join(lines)
