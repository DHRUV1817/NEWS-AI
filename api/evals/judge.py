"""Rubric-scored LLM judgement of summary quality.

This is the least objective family of metrics in the harness and is reported
separately for that reason. An LLM grading an LLM is circular unless it is
calibrated against human scores; the report states the judged numbers and the
calibration coverage side by side rather than presenting them as ground truth.
"""

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from evals.metrics import ExtractionResult
from newsninja.analysis.client import StructuredClient
from newsninja.analysis.grounding import source_corpus

JUDGE_MODEL = "openai/gpt-oss-120b"

JUDGE_SYSTEM = (Path(__file__).parent / "prompts" / "judge.md").read_text(
    encoding="utf-8"
)


class JudgeScore(BaseModel):
    coverage: int = Field(ge=1, le=5)
    neutrality: int = Field(ge=1, le=5)
    coherence: int = Field(ge=1, le=5)
    rationale: str


@dataclass
class JudgedMetrics:
    judged_count: int
    mean_coverage: float | None
    mean_neutrality: float | None
    mean_coherence: float | None


def judge_summary(
    client: StructuredClient, result: ExtractionResult, model: str = JUDGE_MODEL
) -> JudgeScore:
    """Score one topic's summary against its source articles."""
    analysis = result.analysis
    if analysis is None:
        raise ValueError("cannot judge a failed extraction")

    user = (
        f"Topic: {result.topic}\n\n"
        f"Summary under evaluation:\n{analysis.summary}\n\n"
        f"Source articles:\n{source_corpus(result.articles)}"
    )
    return client.structured(
        model=model, system=JUDGE_SYSTEM, user=user, schema_model=JudgeScore
    )


def judged_metrics(
    client: StructuredClient, results: list[ExtractionResult], model: str = JUDGE_MODEL
) -> JudgedMetrics:
    """Judge every successful extraction and average the scores."""
    judgeable = [r for r in results if not r.failed and r.analysis is not None]
    if not judgeable:
        return JudgedMetrics(0, None, None, None)

    scores = [judge_summary(client, result, model=model) for result in judgeable]
    count = len(scores)
    return JudgedMetrics(
        judged_count=count,
        mean_coverage=sum(s.coverage for s in scores) / count,
        mean_neutrality=sum(s.neutrality for s in scores) / count,
        mean_coherence=sum(s.coherence for s in scores) / count,
    )
