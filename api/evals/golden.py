"""Ground-truth labels for the eval corpus.

Labels are drafted by a model and corrected by a human. ``reviewed`` records
which have actually been through human correction; only those count toward any
reported metric. Presenting model-drafted labels as ground truth would make
every agreement number circular.
"""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, model_validator

GOLDEN_PATH = Path(__file__).parent / "data" / "golden.jsonl"

Provenance = Literal["model-drafted", "human-authored", "human-corrected"]


class GoldenLabel(BaseModel):
    topic: str
    entities: list[str]
    stance: Literal["positive", "negative", "neutral"]
    supported_claim_quotes: list[str]
    reviewed: bool = False
    provenance: Provenance = "model-drafted"

    @model_validator(mode="after")
    def _reviewed_cannot_be_model_drafted(self) -> Self:
        """A reviewed label grading itself as still model-drafted would let
        model output become its own ground truth. A human who reviews a
        draft must set ``provenance`` to ``"human-corrected"`` (or author one
        directly as ``"human-authored"``)."""
        if self.reviewed and self.provenance == "model-drafted":
            raise ValueError(
                "reviewed=True is incompatible with provenance='model-drafted'; "
                "set provenance='human-corrected' once a human has checked it"
            )
        return self


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
