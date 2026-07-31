"""Domain models. These double as the LLM output contract via JSON Schema."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

EntityKind = Literal["person", "org", "place", "product", "other"]
Stance = Literal["positive", "negative", "neutral"]


class Article(BaseModel):
    """One fetched item. ``body`` is the text quote-grounding is checked against."""

    title: str
    url: str
    source: str
    body: str = ""
    published: datetime | None = None


class Entity(BaseModel):
    name: str
    kind: EntityKind


class Claim(BaseModel):
    """A factual assertion plus the verbatim span supporting it.

    ``quote`` must appear character-for-character in the source article. This is
    what makes hallucination checkable rather than a matter of opinion.
    """

    text: str
    quote: str


class ArticleAnalysis(BaseModel):
    topic: str
    summary: str
    entities: list[Entity]
    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    key_claims: list[Claim]


class Briefing(BaseModel):
    topics: list[str]
    script: str
    analyses: list[ArticleAnalysis]
    language: str = "en"
