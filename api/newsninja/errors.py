"""Typed errors. Failures are surfaced, never swallowed into user-facing strings."""


class NewsNinjaError(Exception):
    """Base for every error this package raises."""


class SourceError(NewsNinjaError):
    """A source failed to fetch. Carries the source name so the UI can name it."""

    def __init__(self, source: str, message: str) -> None:
        self.source = source
        super().__init__(f"{source}: {message}")


class ExtractionFailure(NewsNinjaError):
    """The model could not produce schema-valid output within the retry budget."""


class RateLimitError(NewsNinjaError):
    """The provider's rate limit was hit. ``retry_after`` is in seconds."""

    def __init__(self, message: str, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(message)
