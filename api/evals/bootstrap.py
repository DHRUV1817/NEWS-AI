"""Draft candidate golden labels for a human to correct.

Drafting is not labelling. Every record produced here is ``reviewed=False`` and
is invisible to metrics until a human flips it. The point is to turn "author 40
labels from scratch" into "correct 40 drafts", which is the difference between
a golden set that exists and one that never gets made.
"""

import argparse
import sys

from evals.corpus import CorpusRecord, load_corpus
from evals.golden import GOLDEN_PATH, GoldenLabel, load_golden, save_golden
from newsninja.analysis.client import StructuredClient
from newsninja.analysis.extract import extract_topic
from newsninja.analysis.grounding import source_corpus

BOOTSTRAP_MODEL = "openai/gpt-oss-120b"


def bootstrap(
    client: StructuredClient,
    records: list[CorpusRecord],
    model: str = BOOTSTRAP_MODEL,
) -> list[GoldenLabel]:
    """Draft one label per corpus record using the strongest available model.

    A record with no articles has nothing for the model to read; asserting a
    stance or entities from no evidence would draft a label from nothing, so
    such records are skipped rather than labelled.
    """
    labels: list[GoldenLabel] = []
    for record in records:
        if not record.articles:
            continue
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


def merge_golden(
    existing: list[GoldenLabel], drafted: list[GoldenLabel]
) -> list[GoldenLabel]:
    """Merge freshly drafted labels into an existing golden set.

    A human-reviewed label is never overwritten by a draft, even if one was
    produced for the same topic — the golden set represents irreplaceable
    manual work, and a second bootstrap run must not silently clobber it.
    Every other topic (absent, or present but still unreviewed) takes the
    freshly drafted label.
    """
    merged: dict[str, GoldenLabel] = {label.topic: label for label in existing}
    for label in drafted:
        current = merged.get(label.topic)
        if current is not None and current.reviewed:
            continue
        merged[label.topic] = label
    return list(merged.values())


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
    existing = load_golden()

    # Never spend a call re-drafting a topic a human has already reviewed.
    reviewed_topics = {label.topic for label in existing if label.reviewed}
    to_draft = [record for record in records if record.topic not in reviewed_topics]

    drafted = bootstrap(GroqClient(api_key=settings.groq_api_key), to_draft)
    merged = merge_golden(existing, drafted)
    save_golden(merged)

    preserved = sum(1 for label in existing if label.reviewed)
    print(f"preserved {preserved} human-reviewed labels, drafted {len(drafted)} labels")
    print(f"-> {GOLDEN_PATH}")
    print(
        "Every drafted label is reviewed=false and counts toward nothing until "
        "you correct it and set reviewed=true.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
