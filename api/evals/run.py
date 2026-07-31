"""Run the evaluation harness and write a report.

Human-invoked; not part of CI. A full run makes real API calls against the
free tier, so it is rate-limited by the same limiter production uses.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from evals.agreement import agreement_metrics
from evals.corpus import CorpusRecord, load_corpus
from evals.golden import load_golden
from evals.judge import judged_metrics
from evals.metrics import ExtractionResult, deterministic_metrics
from evals.report import render_report
from newsninja.analysis.client import StructuredClient
from newsninja.analysis.extract import extract_topic
from newsninja.errors import ExtractionFailure

REPORT_DIR = Path(__file__).resolve().parents[2] / "docs" / "evals"


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

    from newsninja.analysis.client import GroqClient
    from newsninja.analysis.extract import DEFAULT_MODEL
    from newsninja.analysis.prompts import PROMPT_VERSION
    from newsninja.config import Settings

    settings = Settings()
    client = GroqClient(api_key=settings.groq_api_key)
    records = load_corpus()

    results = extract_results(client, records)

    deterministic = deterministic_metrics(results)
    agreement = agreement_metrics(results, load_golden())
    judged = judged_metrics(client, results) if not args.no_judge else judged_metrics(
        client, []
    )

    report = render_report(
        deterministic, agreement, judged,
        {
            "generated": datetime.now(UTC).date().isoformat(),
            "model": DEFAULT_MODEL,
            "prompt_version": PROMPT_VERSION,
        },
    )
    print(report)

    if args.report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / "latest.md"
        path.write_text(report, encoding="utf-8")
        print(f"\nwritten to {path}", file=sys.stderr)

    print(
        f"\ntokens: {client.usage.prompt_tokens} prompt / "
        f"{client.usage.completion_tokens} completion / {client.usage.calls} calls",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
