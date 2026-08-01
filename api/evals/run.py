"""Run the evaluation harness and write a report.

Human-invoked; not part of CI. A full run makes real API calls against the
free tier, so it is rate-limited by the same limiter production uses.

Every check that can reject the inputs runs *before* the first call. Tokens
are the one irreversible thing this script spends, and a malformed golden set
discovered afterwards would abort the run with the budget already gone, the
deterministic metrics already computed and thrown away, and no report written.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from evals.agreement import (
    agreement_metrics,
    duplicate_topic_problem,
    unreviewed_duplicate_topics,
)
from evals.corpus import CorpusRecord, load_corpus
from evals.golden import GoldenLabel, load_golden, reviewed_only
from evals.judge import judged_metrics
from evals.metrics import ExtractionResult, deterministic_metrics
from evals.report import render_report, report_payload
from newsninja.analysis.client import GroqClient, StructuredClient
from newsninja.analysis.extract import extract_topic
from newsninja.errors import ExtractionFailure

REPORT_DIR = Path(__file__).resolve().parents[2] / "docs" / "evals"

EXIT_INVALID_INPUT = 2


def validate_inputs(
    records: list[CorpusRecord], labels: list[GoldenLabel]
) -> list[str]:
    """Every reason the inputs cannot produce trustworthy numbers.

    The same duplicate rules ``agreement_metrics`` enforces, asked here first
    so the answer costs nothing. ``agreement_metrics`` still raises for a
    direct caller; this exists so the runner never learns about a malformed
    input from a traceback thrown after the budget is spent.
    """
    problems = []

    corpus_problem = duplicate_topic_problem(
        (record.topic for record in records), "corpus"
    )
    if corpus_problem:
        problems.append(corpus_problem)

    golden_problem = duplicate_topic_problem(
        (label.topic for label in reviewed_only(labels)), "reviewed golden set"
    )
    if golden_problem:
        problems.append(golden_problem)

    return problems


def input_warnings(labels: list[GoldenLabel]) -> list[str]:
    """Things worth saying out loud that are not worth refusing to run over."""
    duplicates = unreviewed_duplicate_topics(labels)
    if not duplicates:
        return []
    return [
        (
            f"duplicate unreviewed golden labels for: {', '.join(duplicates)}; "
            f"they back no reported metric, but the file looks hand-edited"
        )
    ]


def _build_client() -> GroqClient:
    """Built only once the inputs are known good, so a malformed golden set
    does not even need credentials to be rejected."""
    from newsninja.config import Settings

    return GroqClient(api_key=Settings().groq_api_key)


def extract_results(
    client: StructuredClient, records: list[CorpusRecord]
) -> list[ExtractionResult]:
    """Extract every corpus record that actually has articles.

    A record with no articles is skipped, not recorded. ``extract_topic``
    short-circuits an empty article list to a synthetic analysis — a
    hard-coded ``stance="neutral"``, no entities, no claims — without calling
    the model at all. Recording that as a successful extraction would credit
    the harness for work it never did: it counts toward ``schema_valid_rate``
    as a valid schema nothing produced, and its fabricated ``"neutral"`` scores
    against a reviewed ``neutral`` label for free stance accuracy. Skipping
    keeps every denominator equal to the number of extractions really
    attempted.
    """
    results: list[ExtractionResult] = []
    for record in records:
        if not record.articles:
            print(
                f"skipping {record.topic!r}: no articles captured, nothing to extract",
                file=sys.stderr,
            )
            continue
        try:
            analysis = extract_topic(client, record.topic, record.articles)
            results.append(ExtractionResult(topic=record.topic, analysis=analysis,
                                            articles=record.articles, failed=False))
        except ExtractionFailure as exc:
            print(f"extraction failed for {record.topic!r}: {exc}", file=sys.stderr)
            results.append(ExtractionResult(topic=record.topic, analysis=None,
                                            articles=record.articles, failed=True))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evals.run", description="Run the evaluation harness."
    )
    parser.add_argument("--report", action="store_true", help="Write the markdown report.")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge.")
    args = parser.parse_args(argv)

    from newsninja.analysis.extract import DEFAULT_MODEL
    from newsninja.analysis.prompts import PROMPT_VERSION

    records = load_corpus()
    labels = load_golden()

    for warning in input_warnings(labels):
        print(f"warning: {warning}", file=sys.stderr)

    problems = validate_inputs(records, labels)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        print(
            "aborted before any API call — nothing was spent, nothing to redo.",
            file=sys.stderr,
        )
        return EXIT_INVALID_INPUT

    client = _build_client()
    skipped = [record.topic for record in records if not record.articles]

    results = extract_results(client, records)

    deterministic = deterministic_metrics(results)
    agreement = agreement_metrics(results, labels)
    judged = judged_metrics(client, results) if not args.no_judge else judged_metrics(
        client, []
    )

    meta = {
        "generated": datetime.now(UTC).date().isoformat(),
        "model": DEFAULT_MODEL,
        "prompt_version": PROMPT_VERSION,
        "skipped_records": len(skipped),
    }
    report = render_report(deterministic, agreement, judged, meta)
    print(report)

    if args.report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / "latest.md"
        path.write_text(report, encoding="utf-8")

        # A machine-readable sibling, so anything that displays these numbers
        # reads a contract rather than parsing prose that changes whenever the
        # wording improves.
        payload = report_payload(deterministic, agreement, judged, meta)
        data_path = REPORT_DIR / "latest.json"
        data_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwritten to {path} and {data_path}", file=sys.stderr)

    print(
        f"\ntokens: {client.usage.prompt_tokens} prompt / "
        f"{client.usage.completion_tokens} completion / {client.usage.calls} calls",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
