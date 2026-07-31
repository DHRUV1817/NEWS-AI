"""HTTP request and response bodies.

Kept apart from ``newsninja.models``: those are the domain contract and the
LLM output schema, and letting HTTP concerns leak into them would couple the
wire format to the thing being measured.
"""

from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

from newsninja.audio.tts import SUPPORTED_LANGUAGES
from newsninja.models import ArticleAnalysis, Briefing
from newsninja.pipeline import MAX_TOPICS

#: A language code reaches a translation prompt, so it is not free text.
LANGUAGE_PATTERN = r"^[a-z]{2}(-[A-Za-z]{2})?$"


def _supported_language(value: str) -> str:
    """Refuse a code the pipeline would silently rewrite.

    The shape check is not enough. ``"sv"`` and ``"en-GB"`` both match the
    pattern, and ``synthesize_speech`` rewrites anything outside
    ``SUPPORTED_LANGUAGES`` to ``"en"`` without saying so — so a caller asking
    for Swedish gets a Swedish script spoken in English, with nothing in the
    response indicating that happened. Refusing is the honest answer; the set
    lives in ``audio/tts.py`` because that is what enforces it downstream.
    """
    if value not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"language {value!r} is not supported; "
            f"choose one of {sorted(SUPPORTED_LANGUAGES)}"
        )
    return value


#: Shape first, then membership. The shape check is what keeps a rejected code
#: out of an error message unescaped; the membership check is what keeps the
#: caller from being told nothing about a language the pipeline cannot speak.
Language = Annotated[
    str, Field(pattern=LANGUAGE_PATTERN), AfterValidator(_supported_language)
]

#: chunk_text splits at 3,000 characters and gTTS makes one outbound call per
#: chunk, so an uncapped script is an amplification vector: 1 MB of text becomes
#: roughly 350 requests leaving the host.
MAX_SCRIPT_CHARS = 20_000

#: Total caller-supplied text one /brief request may carry across all its
#: analyses. Bounding the count without bounding the size bounds nothing:
#: measured, four analyses with 6,000-character summaries reserve 7,737 of
#: gpt-oss-120b's 8,000 TPM — 97% of the shared minute, from one unauthenticated
#: request carrying analyses the server never produced. The per-IP window is no
#: help there: one request a minute holds the whole budget at zero.
#:
#: GroqClient estimates four characters per token and adds
#: EXPECTED_COMPLETION_TOKENS on top, so 8,000 characters reserves roughly
#: 8000/4 + 1500 = 3,500 tokens — under half the minute, which leaves a second
#: caller served rather than 429ed. It is also generous against what /analyze
#: actually produces: 1,600 characters per analysis at the five-analysis
#: maximum. test_a_maximal_brief_request_reserves_under_half_the_budget pins
#: the arithmetic so this comment cannot drift away from the code.
MAX_BRIEF_CHARS = 8_000


def _text_chars(analysis: ArticleAnalysis) -> int:
    """Free text in one analysis — every field the caller writes.

    Wider than what reaches the synthesis prompt (which renders topic, summary,
    entity names and claim texts, but not quotes) on purpose: the quote is still
    caller-supplied text the server holds in memory, and a bound that covers
    less than the request does not bound the request.
    """
    return (
        len(analysis.topic)
        + len(analysis.summary)
        + sum(len(entity.name) for entity in analysis.entities)
        + sum(len(claim.text) + len(claim.quote) for claim in analysis.key_claims)
    )


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
    language: Language = "en"

    @model_validator(mode="after")
    def _bound_the_total_text(self) -> Self:
        """The count bound is per-request; this is the size bound behind it."""
        total = sum(_text_chars(analysis) for analysis in self.analyses)
        if total > MAX_BRIEF_CHARS:
            raise ValueError(
                f"analyses carry {total} characters of text; at most "
                f"{MAX_BRIEF_CHARS} are accepted across one request"
            )
        return self


class BriefResponse(BaseModel):
    briefing: Briefing


class AudioRequest(BaseModel):
    script: str = Field(min_length=1, max_length=MAX_SCRIPT_CHARS)
    language: Language = "en"


class HealthResponse(BaseModel):
    status: str
    version: str
