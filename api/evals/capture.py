"""Capture articles into the eval corpus. Run by a human, not by CI."""

import argparse
import sys

from evals.corpus import CORPUS_PATH, CorpusRecord, save_corpus
from newsninja.config import Settings
from newsninja.errors import SourceError
from newsninja.sources.base import Source
from newsninja.sources.google_news import GoogleNewsSource

DEFAULT_TOPICS = [
    "artificial intelligence",
    "climate change",
    "cryptocurrency",
    "space exploration",
    "renewable energy",
]


def capture(topics: list[str], source: Source, limit: int = 8) -> list[CorpusRecord]:
    """Fetch articles for each topic. A topic whose fetch fails is skipped."""
    records: list[CorpusRecord] = []
    for topic in topics:
        try:
            articles = source.fetch(topic, limit=limit)
        except SourceError as exc:
            print(f"skipping {topic!r}: {exc}", file=sys.stderr)
            continue
        records.append(CorpusRecord(topic=topic, articles=articles))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evals.capture", description="Capture articles into the eval corpus."
    )
    parser.add_argument("--topic", action="append", help="Topic. Repeatable.")
    parser.add_argument("--limit", type=int, default=8, help="Articles per topic.")
    args = parser.parse_args(argv)

    settings = Settings()
    topics = args.topic or DEFAULT_TOPICS
    records = capture(
        topics,
        GoogleNewsSource(timeout=settings.request_timeout),
        limit=args.limit,
    )
    if not records:
        print("captured nothing; corpus not written", file=sys.stderr)
        return 1

    save_corpus(records)
    total = sum(len(r.articles) for r in records)
    print(f"captured {total} articles across {len(records)} topics -> {CORPUS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
