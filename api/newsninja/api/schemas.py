"""HTTP request and response bodies.

Kept apart from ``newsninja.models``: those are the domain contract and the
LLM output schema, and letting HTTP concerns leak into them would couple the
wire format to the thing being measured.
"""

from pydantic import BaseModel, Field, field_validator

from newsninja.models import ArticleAnalysis, Briefing
from newsninja.pipeline import MAX_TOPICS

#: A language code reaches a translation prompt, so it is not free text.
LANGUAGE_PATTERN = r"^[a-z]{2}(-[A-Za-z]{2})?$"

#: chunk_text splits at 3,000 characters and gTTS makes one outbound call per
#: chunk, so an uncapped script is an amplification vector: 1 MB of text becomes
#: roughly 350 requests leaving the host.
MAX_SCRIPT_CHARS = 20_000


class AnalyzeRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=8, ge=1, le=12)

    @field_validator("topic")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("topic must not be blank")
        return stripped


class AnalyzeResponse(BaseModel):
    analysis: ArticleAnalysis
    #: Sources that were tried and broke, by name.
    source_errors: dict[str, list[str]] = {}
    #: Sources that reported themselves unavailable and were never tried.
    #: Separate from source_errors on purpose: a missing credential is
    #: something the caller can fix, a broken source is not.
    skipped_sources: list[str] = []


class BriefRequest(BaseModel):
    analyses: list[ArticleAnalysis] = Field(min_length=1, max_length=MAX_TOPICS)
    language: str = Field(default="en", pattern=LANGUAGE_PATTERN)


class BriefResponse(BaseModel):
    briefing: Briefing


class AudioRequest(BaseModel):
    script: str = Field(min_length=1, max_length=MAX_SCRIPT_CHARS)
    language: str = Field(default="en", pattern=LANGUAGE_PATTERN)


class HealthResponse(BaseModel):
    status: str
    version: str
