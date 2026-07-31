"""Run the evaluation harness and write a report.

Human-invoked; not part of CI. A full run makes real API calls against the
free tier, so it is rate-limited by the same limiter production uses.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from evals.agreement import agreement_metrics
from evals.corpus import load_corpus
from evals.golden import load_golden
from evals.judge import judged_metrics
from evals.metrics import ExtractionResult, deterministic_metrics
from evals.report import render_report

REPORT_DIR = Path(__file__).resolve().parents[2] / "docs" / "evals"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evals.run", description="Run the evaluation harness."
    )
    parser.add_argument("--report", action="store_true", help="Write the markdown report.")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge.")
    args = parser.parse_args(argv)

    from newsninja.analysis.client import GroqClient
    from newsninja.analysis.extract import DEFAULT_MODEL, extract_topic
    from newsninja.analysis.prompts import PROMPT_VERSION
    from newsninja.config import Settings
    from newsninja.errors import ExtractionFailure

    settings = Settings()
    client = GroqClient(api_key=settings.groq_api_key)
    records = load_corpus()

    results: list[ExtractionResult] = []
    for record in records:
        try:
            analysis = extract_topic(client, record.topic, record.articles)
            results.append(ExtractionResult(topic=record.topic, analysis=analysis,
                                            articles=record.articles, failed=False))
        except ExtractionFailure as exc:
            print(f"extraction failed for {record.topic!r}: {exc}", file=sys.stderr)
            results.append(ExtractionResult(topic=record.topic, analysis=None,
                                            articles=record.articles, failed=True))

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
