"""Ground-truth labels for the eval corpus.

Labels are drafted by a model and corrected by a human. ``reviewed`` records
which have actually been through human correction; only those count toward any
reported metric. Presenting model-drafted labels as ground truth would make
every agreement number circular.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

GOLDEN_PATH = Path(__file__).parent / "data" / "golden.jsonl"

Provenance = Literal["model-drafted", "human-authored", "human-corrected"]


class GoldenLabel(BaseModel):
    topic: str
    entities: list[str]
    stance: Literal["positive", "negative", "neutral"]
    supported_claim_quotes: list[str]
    reviewed: bool = False
    provenance: Provenance = "model-drafted"


def save_golden(labels: list[GoldenLabel], path: Path = GOLDEN_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(label.model_dump_json() + "\n")


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenLabel]:
    """Load labels. A missing file is an empty set, not an error — the harness
    runs its label-free metrics with no golden set at all."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [
            GoldenLabel.model_validate_json(line) for line in handle if line.strip()
        ]


def reviewed_only(labels: list[GoldenLabel]) -> list[GoldenLabel]:
    """Only human-reviewed labels may back a reported metric."""
    return [label for label in labels if label.reviewed]
