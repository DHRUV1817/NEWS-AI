"""Draft candidate golden labels for a human to correct.

Drafting is not labelling. Every record produced here is ``reviewed=False`` and
is invisible to metrics until a human flips it. The point is to turn "author 40
labels from scratch" into "correct 40 drafts", which is the difference between
a golden set that exists and one that never gets made.
"""

import argparse
import sys

from evals.corpus import CorpusRecord, load_corpus
from evals.golden import GOLDEN_PATH, GoldenLabel, save_golden
from newsninja.analysis.client import StructuredClient
from newsninja.analysis.extract import extract_topic
from newsninja.analysis.grounding import source_corpus

BOOTSTRAP_MODEL = "openai/gpt-oss-120b"


def bootstrap(
    client: StructuredClient,
    records: list[CorpusRecord],
    model: str = BOOTSTRAP_MODEL,
) -> list[GoldenLabel]:
    """Draft one label per corpus record using the strongest available model."""
    labels: list[GoldenLabel] = []
    for record in records:
        analysis = extract_topic(client, record.topic, record.articles, model=model)
        corpus = source_corpus(record.articles)
        labels.append(
            GoldenLabel(
                topic=record.topic,
                entities=[entity.name for entity in analysis.entities],
                stance=analysis.stance,
                # A drafted quote that is not actually in the source is already
                # known-wrong; do not seed it into ground truth.
                supported_claim_quotes=[
                    claim.quote for claim in analysis.key_claims if claim.quote in corpus
                ],
                reviewed=False,
                provenance="model-drafted",
            )
        )
    return labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evals.bootstrap",
        description="Draft golden labels for human correction.",
    )
    parser.parse_args(argv)

    from newsninja.analysis.client import GroqClient
    from newsninja.config import Settings

    settings = Settings()
    records = load_corpus()
    labels = bootstrap(GroqClient(api_key=settings.groq_api_key), records)
    save_golden(labels)

    print(f"drafted {len(labels)} labels -> {GOLDEN_PATH}")
    print(
        "Every label is reviewed=false and counts toward nothing until you "
        "correct it and set reviewed=true.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
