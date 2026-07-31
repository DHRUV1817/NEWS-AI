"""Captured articles used as a stable substrate for evaluation.

Captured once and committed so every run measures the model rather than
whatever the news happened to be that morning.
"""

from pathlib import Path

from pydantic import BaseModel

from newsninja.models import Article

CORPUS_PATH = Path(__file__).parent / "data" / "corpus.jsonl"


class CorpusRecord(BaseModel):
    """One topic and the articles captured for it."""

    topic: str
    articles: list[Article]


def save_corpus(records: list[CorpusRecord], path: Path = CORPUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def load_corpus(path: Path = CORPUS_PATH) -> list[CorpusRecord]:
    if not path.exists():
        raise FileNotFoundError(
            f"No corpus at {path}. Run `python -m evals.capture` to capture one."
        )
    with path.open(encoding="utf-8") as handle:
        return [
            CorpusRecord.model_validate_json(line)
            for line in handle
            if line.strip()
        ]
