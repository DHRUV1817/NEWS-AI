"""The seam every source implements.

Adding a source is a new file, never an edit to the pipeline.
"""

from typing import Protocol, runtime_checkable

from newsninja.models import Article


@runtime_checkable
class Source(Protocol):
    name: str

    def available(self) -> bool:
        """False when required credentials are missing. The UI reports this
        as unavailable rather than as a failure."""

    def fetch(self, topic: str, limit: int = 8) -> list[Article]:
        """Fetch up to ``limit`` articles. Raises SourceError on transport failure."""
