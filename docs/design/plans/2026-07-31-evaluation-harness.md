# NewsNinja Evaluation Harness — Implementation Plan

> Execute this plan task by task, reviewing each task before starting the next.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure what the extraction pipeline actually produces — schema validity, quote grounding, hallucination rate, entity precision/recall, stance agreement, and judged summary quality — and emit a committed report the README can cite.

**Architecture:** An `evals/` package beside `newsninja/`, importing it but never imported by it. Metrics are split by how objective they are: deterministic checks need no labels and run today; agreement metrics need a human-reviewed golden set; judged metrics use an LLM against a rubric and are reported separately with a calibration figure. A runner ties them together and writes markdown to `docs/evals/`.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, Groq (`openai/gpt-oss-120b` for judging). No new runtime dependencies — Cohen's κ is hand-rolled rather than pulling in scipy.

This is **Plan 2 of 4**. Plan 1 (core package) is complete on `rebuild/core-package`. Source spec: `docs/design/specs/2026-07-31-newsninja-portfolio-design.md` §6.

## Global Constraints

- **Python 3.12** via `uv run --python 3.12 --extra dev ...` from inside `api/`. System Python is 3.9.6 and cannot run the dependencies.
- **All work in `api/evals/` and `api/tests/`.** Never modify `newsninja/` — the harness measures it, it does not change it. If a metric needs something `newsninja` does not expose, stop and report rather than editing the package.
- **`evals/` imports `newsninja`. `newsninja` must never import `evals`.**
- **No network in tests.** The suite must stay offline and free. Live-API work happens only when a human runs the runner explicitly.
- **mypy `strict = true` with `plugins = ["pydantic.mypy"]`. ZERO `type: ignore`.** The package currently has exactly one `# noqa` (`BLE001, S110` on the Orpheus fallback) — do not add more.
- **Never fabricate a metric.** A metric that cannot be computed reports `unavailable` with a reason. It never reports `0.0`, and never reports a number derived from unreviewed labels.
- **Commit messages carry no AI attribution** — no `Co-Authored-By`, no "Generated with" footer.

## The labelling problem, and how this plan handles it

Entity precision/recall and stance agreement need ground truth only a human can supply. The spec asks for ~40 hand-labelled articles; that is real work and it is the single thing blocking the project's strongest claim.

This plan does three things about it:

1. **Most metrics need no labels.** Schema validity, quote grounding, hallucination rate, retry counts, and token cost are all computable today against live output. Those ship first (Task 3) and give the README real numbers immediately.
2. **The golden set gets drafted, not authored from scratch.** Task 4 builds a bootstrapper that runs `gpt-oss-120b` over captured articles and writes candidate labels with `reviewed: false`. The human corrects rather than composes.
3. **Unreviewed labels never count.** Every label-dependent metric filters on `reviewed: true` and reports coverage (`18/40 reviewed`). A fully unreviewed golden set yields `unavailable`, not a flattering number.

The provenance is recorded in the report and the README: model-drafted, human-corrected. Presenting model-drafted labels as ground truth would be the same category of dishonesty this whole rebuild exists to remove.

---

### Task 1: Eval package scaffold and article capture

Real articles are needed as a stable substrate. Captured once, committed, and reused so every eval run measures the model rather than today's news.

**Files:**
- Create: `api/evals/__init__.py`, `api/evals/corpus.py`, `api/tests/test_evals_corpus.py`
- Create: `api/evals/data/.gitkeep`

**Interfaces:**
- Consumes: `newsninja.models.Article`, `newsninja.sources.google_news.GoogleNewsSource`
- Produces: `CorpusRecord(topic, articles)`, `load_corpus(path) -> list[CorpusRecord]`, `save_corpus(records, path) -> None`, `CORPUS_PATH`

- [ ] **Step 1: Write the failing test**

`api/tests/test_evals_corpus.py`:

```python
import pytest

from evals.corpus import CorpusRecord, load_corpus, save_corpus
from newsninja.models import Article


def _record(topic: str = "ai") -> CorpusRecord:
    return CorpusRecord(
        topic=topic,
        articles=[
            Article(title="A", url="https://e.com/1", source="google_news", body="body one"),
            Article(title="B", url="https://e.com/2", source="google_news", body="body two"),
        ],
    )


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "corpus.jsonl"
    save_corpus([_record("ai"), _record("climate")], path)
    loaded = load_corpus(path)
    assert [r.topic for r in loaded] == ["ai", "climate"]
    assert loaded[0].articles[0].body == "body one"


def test_load_missing_file_raises_with_a_usable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="capture"):
        load_corpus(tmp_path / "absent.jsonl")


def test_save_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "corpus.jsonl"
    save_corpus([_record("ai"), _record("climate")], path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2


def test_empty_corpus_round_trips(tmp_path):
    path = tmp_path / "corpus.jsonl"
    save_corpus([], path)
    assert load_corpus(path) == []
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_corpus.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evals'`

- [ ] **Step 3: Write the implementation**

```bash
mkdir -p api/evals/data && touch api/evals/__init__.py api/evals/data/.gitkeep
```

`api/evals/corpus.py`:

```python
"""Captured articles used as a stable substrate for evaluation.

Captured once and committed so every run measures the model rather than
whatever the news happened to be that morning.
"""

import json
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_corpus.py -v
```

Expected: 4 passed

- [ ] **Step 5: Verify the three gates, then commit**

```bash
cd api && uv run --python 3.12 --extra dev pytest -q \
  && uv run --python 3.12 --extra dev ruff check . \
  && uv run --python 3.12 --extra dev mypy newsninja evals
cd .. && git add api/evals api/tests/test_evals_corpus.py
git commit -m "Add eval corpus storage

Captured articles are committed so every eval run measures the model rather
than whatever the news happened to be that morning."
```

---

### Task 2: Corpus capture command

**Files:**
- Create: `api/evals/capture.py`, `api/tests/test_evals_capture.py`

**Interfaces:**
- Consumes: `CorpusRecord`, `save_corpus`, `GoogleNewsSource`
- Produces: `capture(topics, source, limit=8) -> list[CorpusRecord]`, `main(argv=None) -> int`

- [ ] **Step 1: Write the failing test**

`api/tests/test_evals_capture.py`:

```python
from evals.capture import capture
from newsninja.errors import SourceError
from newsninja.models import Article


class FakeSource:
    name = "google_news"

    def __init__(self, error=None):
        self._error = error
        self.calls: list[str] = []

    def available(self) -> bool:
        return True

    def fetch(self, topic, limit=8):
        self.calls.append(topic)
        if self._error:
            raise self._error
        return [
            Article(title=f"{topic} {i}", url=f"https://e.com/{topic}/{i}",
                    source="google_news", body=f"body {i}")
            for i in range(limit)
        ]


def test_capture_returns_one_record_per_topic():
    records = capture(["ai", "climate"], source=FakeSource(), limit=3)
    assert [r.topic for r in records] == ["ai", "climate"]
    assert all(len(r.articles) == 3 for r in records)


def test_capture_skips_a_topic_whose_fetch_fails():
    source = FakeSource(error=SourceError("google_news", "boom"))
    records = capture(["ai"], source=source, limit=3)
    assert records == []


def test_capture_records_are_independent_per_topic():
    records = capture(["ai", "climate"], source=FakeSource(), limit=2)
    assert records[0].articles[0].title.startswith("ai")
    assert records[1].articles[0].title.startswith("climate")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_capture.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evals.capture'`

- [ ] **Step 3: Write the implementation**

`api/evals/capture.py`:

```python
"""Capture articles into the eval corpus. Run by a human, not by CI."""

import argparse
import sys

from evals.corpus import CORPUS_PATH, CorpusRecord, save_corpus
from newsninja.config import Settings
from newsninja.errors import SourceError
from newsninja.sources.base import Source
from newsninja.sources.google_news import GoogleNewsSource

DEFAULT_TOPICS = [
    "artificial intelligence",
    "climate change",
    "cryptocurrency",
    "space exploration",
    "renewable energy",
]


def capture(topics: list[str], source: Source, limit: int = 8) -> list[CorpusRecord]:
    """Fetch articles for each topic. A topic whose fetch fails is skipped."""
    records: list[CorpusRecord] = []
    for topic in topics:
        try:
            articles = source.fetch(topic, limit=limit)
        except SourceError as exc:
            print(f"skipping {topic!r}: {exc}", file=sys.stderr)
            continue
        records.append(CorpusRecord(topic=topic, articles=articles))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evals.capture", description="Capture articles into the eval corpus."
    )
    parser.add_argument("--topic", action="append", help="Topic. Repeatable.")
    parser.add_argument("--limit", type=int, default=8, help="Articles per topic.")
    args = parser.parse_args(argv)

    settings = Settings()
    topics = args.topic or DEFAULT_TOPICS
    records = capture(
        topics,
        GoogleNewsSource(timeout=settings.request_timeout),
        limit=args.limit,
    )
    if not records:
        print("captured nothing; corpus not written", file=sys.stderr)
        return 1

    save_corpus(records)
    total = sum(len(r.articles) for r in records)
    print(f"captured {total} articles across {len(records)} topics -> {CORPUS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_capture.py -v
```

Expected: 3 passed

- [ ] **Step 5: Capture a real corpus**

This step uses the network deliberately — it is a human-run capture, not a test.

```bash
cd api && uv run --python 3.12 --extra dev python -m evals.capture --limit 8
wc -l evals/data/corpus.jsonl
```

Expected: `corpus.jsonl` with 5 lines (one per default topic) and ~40 articles total. If fewer than 3 topics captured, re-run — Google News occasionally rate-limits.

- [ ] **Step 6: Verify gates and commit (including the corpus)**

```bash
cd api && uv run --python 3.12 --extra dev pytest -q \
  && uv run --python 3.12 --extra dev ruff check . \
  && uv run --python 3.12 --extra dev mypy newsninja evals
cd .. && git add api/evals/capture.py api/tests/test_evals_capture.py api/evals/data/corpus.jsonl
git commit -m "Add corpus capture command and capture ~40 real articles

The corpus is committed so eval runs are reproducible against a fixed
substrate rather than live news."
```

---

### Task 3: Label-free deterministic metrics

These need no ground truth and are the metrics the README can cite immediately.

**Files:**
- Create: `api/evals/metrics.py`, `api/tests/test_evals_metrics.py`

**Interfaces:**
- Consumes: `newsninja.models.{Article, ArticleAnalysis}`, `newsninja.analysis.grounding.{grounding_rate, ungrounded_claims}`
- Produces: `DeterministicMetrics(schema_valid_rate, grounding_rate, hallucinated_claim_rate, mean_claims_per_topic, mean_entities_per_topic, topics_evaluated)`, `deterministic_metrics(results) -> DeterministicMetrics`, `ExtractionResult(topic, analysis, articles, failed)`

- [ ] **Step 1: Write the failing test**

`api/tests/test_evals_metrics.py`:

```python
import pytest

from evals.metrics import ExtractionResult, deterministic_metrics
from newsninja.models import Article, ArticleAnalysis, Claim, Entity


def _article(body: str) -> Article:
    return Article(title="t", url="u", source="google_news", body=body)


def _analysis(quotes: list[str], entities: int = 1) -> ArticleAnalysis:
    return ArticleAnalysis(
        topic="ai",
        summary="s",
        entities=[Entity(name=f"E{i}", kind="org") for i in range(entities)],
        stance="neutral",
        confidence=0.5,
        key_claims=[Claim(text=f"c{i}", quote=q) for i, q in enumerate(quotes)],
    )


def test_all_quotes_grounded_gives_rate_one():
    result = ExtractionResult(
        topic="ai", analysis=_analysis(["alpha", "beta"]),
        articles=[_article("alpha and beta appear here")], failed=False,
    )
    m = deterministic_metrics([result])
    assert m.grounding_rate == 1.0
    assert m.hallucinated_claim_rate == 0.0


def test_a_fabricated_quote_is_counted_as_hallucinated():
    result = ExtractionResult(
        topic="ai", analysis=_analysis(["alpha", "never said this"]),
        articles=[_article("alpha appears here")], failed=False,
    )
    m = deterministic_metrics([result])
    assert m.grounding_rate == 0.5
    assert m.hallucinated_claim_rate == 0.5


def test_schema_valid_rate_counts_failed_extractions():
    ok = ExtractionResult(topic="a", analysis=_analysis(["x"]),
                          articles=[_article("x")], failed=False)
    bad = ExtractionResult(topic="b", analysis=None, articles=[_article("y")], failed=True)
    m = deterministic_metrics([ok, bad])
    assert m.schema_valid_rate == 0.5
    assert m.topics_evaluated == 2


def test_grounding_ignores_failed_extractions_rather_than_scoring_them_zero():
    ok = ExtractionResult(topic="a", analysis=_analysis(["x"]),
                          articles=[_article("x")], failed=False)
    bad = ExtractionResult(topic="b", analysis=None, articles=[_article("y")], failed=True)
    m = deterministic_metrics([ok, bad])
    assert m.grounding_rate == 1.0, "a failed extraction is not an ungrounded one"


def test_empty_input_reports_unavailable_not_zero():
    m = deterministic_metrics([])
    assert m.grounding_rate is None
    assert m.schema_valid_rate is None
    assert m.topics_evaluated == 0


def test_mean_counts_are_reported():
    result = ExtractionResult(topic="ai", analysis=_analysis(["a", "b"], entities=3),
                              articles=[_article("a b")], failed=False)
    m = deterministic_metrics([result])
    assert m.mean_claims_per_topic == 2.0
    assert m.mean_entities_per_topic == 3.0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_metrics.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evals.metrics'`

- [ ] **Step 3: Write the implementation**

`api/evals/metrics.py`:

```python
"""Metrics that need no ground truth.

Everything here is arithmetic over the model's own output and its source
articles. No model judges any of it, which is what makes these the numbers
the README can cite without qualification.

A metric that cannot be computed is ``None``, never ``0.0`` — reporting a
zero for "no data" would understate quality as confidently as inventing a
number would overstate it.
"""

from dataclasses import dataclass

from pydantic import BaseModel

from newsninja.analysis.grounding import ungrounded_claims
from newsninja.models import Article, ArticleAnalysis


class ExtractionResult(BaseModel):
    """One topic's extraction outcome, successful or not."""

    topic: str
    analysis: ArticleAnalysis | None
    articles: list[Article]
    failed: bool


@dataclass
class DeterministicMetrics:
    topics_evaluated: int
    schema_valid_rate: float | None
    grounding_rate: float | None
    hallucinated_claim_rate: float | None
    mean_claims_per_topic: float | None
    mean_entities_per_topic: float | None


def deterministic_metrics(results: list[ExtractionResult]) -> DeterministicMetrics:
    """Compute label-free metrics over extraction results."""
    if not results:
        return DeterministicMetrics(0, None, None, None, None, None)

    succeeded = [r for r in results if not r.failed and r.analysis is not None]
    schema_valid_rate = len(succeeded) / len(results)

    if not succeeded:
        return DeterministicMetrics(
            len(results), schema_valid_rate, None, None, None, None
        )

    total_claims = 0
    total_ungrounded = 0
    total_entities = 0
    for result in succeeded:
        analysis = result.analysis
        assert analysis is not None  # narrowed by the filter above
        total_claims += len(analysis.key_claims)
        total_ungrounded += len(ungrounded_claims(analysis, result.articles))
        total_entities += len(analysis.entities)

    grounding = None if total_claims == 0 else 1.0 - (total_ungrounded / total_claims)
    hallucinated = None if total_claims == 0 else total_ungrounded / total_claims

    return DeterministicMetrics(
        topics_evaluated=len(results),
        schema_valid_rate=schema_valid_rate,
        grounding_rate=grounding,
        hallucinated_claim_rate=hallucinated,
        mean_claims_per_topic=total_claims / len(succeeded),
        mean_entities_per_topic=total_entities / len(succeeded),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_metrics.py -v
```

Expected: 6 passed

- [ ] **Step 5: Verify gates and commit**

```bash
cd api && uv run --python 3.12 --extra dev pytest -q \
  && uv run --python 3.12 --extra dev ruff check . \
  && uv run --python 3.12 --extra dev mypy newsninja evals
cd .. && git add api/evals/metrics.py api/tests/test_evals_metrics.py
git commit -m "Add label-free deterministic eval metrics

Schema validity, quote grounding and hallucination rate need no ground truth,
so they can be cited immediately. Unavailable metrics report None rather than
0.0, which would understate quality as confidently as a fabricated number
would overstate it."
```

---

### Task 4: Golden set schema and bootstrapper

**Files:**
- Create: `api/evals/golden.py`, `api/evals/bootstrap.py`, `api/tests/test_evals_golden.py`

**Interfaces:**
- Consumes: `CorpusRecord`, `newsninja.analysis.client.StructuredClient`
- Produces: `GoldenLabel(topic, entities, stance, supported_claim_quotes, reviewed, provenance)`, `load_golden(path)`, `save_golden(labels, path)`, `GOLDEN_PATH`, `reviewed_only(labels)`, `bootstrap(client, records) -> list[GoldenLabel]`

- [ ] **Step 1: Write the failing test**

`api/tests/test_evals_golden.py`:

```python
from evals.golden import GoldenLabel, load_golden, reviewed_only, save_golden
from evals.bootstrap import bootstrap
from evals.corpus import CorpusRecord
from newsninja.models import Article, ArticleAnalysis, Claim, Entity


def _label(topic="ai", reviewed=False):
    return GoldenLabel(
        topic=topic, entities=["OpenAI"], stance="positive",
        supported_claim_quotes=["a quote"], reviewed=reviewed,
        provenance="model-drafted",
    )


def test_labels_default_to_unreviewed():
    assert GoldenLabel(topic="ai", entities=[], stance="neutral",
                       supported_claim_quotes=[]).reviewed is False


def test_round_trips(tmp_path):
    path = tmp_path / "golden.jsonl"
    save_golden([_label("ai"), _label("climate")], path)
    assert [l.topic for l in load_golden(path)] == ["ai", "climate"]


def test_reviewed_only_filters_out_drafts():
    labels = [_label("ai", reviewed=True), _label("climate", reviewed=False)]
    assert [l.topic for l in reviewed_only(labels)] == ["ai"]


def test_load_missing_returns_empty_rather_than_raising(tmp_path):
    assert load_golden(tmp_path / "absent.jsonl") == []


class StubClient:
    def structured(self, *, model, system, user, schema_model, max_retries=2):
        return ArticleAnalysis(
            topic="ai", summary="s",
            entities=[Entity(name="OpenAI", kind="org")],
            stance="positive", confidence=0.9,
            key_claims=[Claim(text="c", quote="models improved")],
        )


def test_bootstrap_marks_every_draft_unreviewed():
    record = CorpusRecord(topic="ai", articles=[
        Article(title="t", url="u", source="google_news", body="models improved")])
    labels = bootstrap(StubClient(), [record])
    assert len(labels) == 1
    assert labels[0].reviewed is False
    assert labels[0].provenance == "model-drafted"
    assert labels[0].entities == ["OpenAI"]


def test_bootstrap_only_keeps_quotes_that_are_actually_in_the_source():
    record = CorpusRecord(topic="ai", articles=[
        Article(title="t", url="u", source="google_news", body="something else entirely")])
    labels = bootstrap(StubClient(), [record])
    assert labels[0].supported_claim_quotes == [], (
        "a drafted quote absent from the source must not become ground truth"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_golden.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evals.golden'`

- [ ] **Step 3: Write the implementation**

`api/evals/golden.py`:

```python
"""Ground-truth labels for the eval corpus.

Labels are drafted by a model and corrected by a human. ``reviewed`` records
which have actually been through human correction; only those count toward any
reported metric. Presenting model-drafted labels as ground truth would make
every agreement number circular.
"""

import json
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
```

`api/evals/bootstrap.py`:

```python
"""Draft candidate golden labels for a human to correct.

Drafting is not labelling. Every record produced here is ``reviewed=False`` and
is invisible to metrics until a human flips it. The point is to turn "author 40
labels from scratch" into "correct 40 drafts", which is the difference between
a golden set that exists and one that never gets made.
"""

import argparse
import sys

from evals.corpus import CorpusRecord, load_corpus
from evals.golden import GOLDEN_PATH, GoldenLabel, save_golden
from newsninja.analysis.client import StructuredClient
from newsninja.analysis.extract import DEFAULT_MODEL, extract_topic
from newsninja.analysis.grounding import source_corpus

BOOTSTRAP_MODEL = "openai/gpt-oss-120b"


def bootstrap(
    client: StructuredClient,
    records: list[CorpusRecord],
    model: str = BOOTSTRAP_MODEL,
) -> list[GoldenLabel]:
    """Draft one label per corpus record using the strongest available model."""
    labels: list[GoldenLabel] = []
    for record in records:
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
    labels = bootstrap(GroqClient(api_key=settings.groq_api_key), records)
    save_golden(labels)

    print(f"drafted {len(labels)} labels -> {GOLDEN_PATH}")
    print(
        "Every label is reviewed=false and counts toward nothing until you "
        "correct it and set reviewed=true.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: `DEFAULT_MODEL` is imported for reference but `BOOTSTRAP_MODEL` is used — drafting deserves the stronger model even though production extraction uses the cheaper one. If ruff flags the unused import, remove it.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_golden.py -v
```

Expected: 7 passed

- [ ] **Step 5: Verify gates and commit**

```bash
cd api && uv run --python 3.12 --extra dev pytest -q \
  && uv run --python 3.12 --extra dev ruff check . \
  && uv run --python 3.12 --extra dev mypy newsninja evals
cd .. && git add api/evals/golden.py api/evals/bootstrap.py api/tests/test_evals_golden.py
git commit -m "Add golden-set schema and a bootstrapper for human correction

Drafted labels are always reviewed=false and count toward nothing until a
human corrects them, so no agreement metric can be backed by model output
grading model output."
```

---

### Task 5: Label-dependent agreement metrics

**Files:**
- Create: `api/evals/agreement.py`, `api/tests/test_evals_agreement.py`

**Interfaces:**
- Consumes: `GoldenLabel`, `ExtractionResult`
- Produces: `AgreementMetrics(labelled_coverage, entity_precision, entity_recall, entity_f1, stance_accuracy, stance_kappa, unavailable_reason)`, `agreement_metrics(results, labels) -> AgreementMetrics`, `cohens_kappa(a, b) -> float | None`

- [ ] **Step 1: Write the failing test**

`api/tests/test_evals_agreement.py`:

```python
import pytest

from evals.agreement import agreement_metrics, cohens_kappa
from evals.golden import GoldenLabel
from evals.metrics import ExtractionResult
from newsninja.models import Article, ArticleAnalysis, Entity


def _result(topic, entities, stance):
    return ExtractionResult(
        topic=topic,
        analysis=ArticleAnalysis(
            topic=topic, summary="s",
            entities=[Entity(name=e, kind="org") for e in entities],
            stance=stance, confidence=0.5, key_claims=[],
        ),
        articles=[Article(title="t", url="u", source="google_news", body="b")],
        failed=False,
    )


def _label(topic, entities, stance, reviewed=True):
    return GoldenLabel(topic=topic, entities=entities, stance=stance,
                       supported_claim_quotes=[], reviewed=reviewed,
                       provenance="human-corrected")


def test_perfect_entity_match_scores_one():
    m = agreement_metrics([_result("ai", ["A", "B"], "positive")],
                          [_label("ai", ["A", "B"], "positive")])
    assert m.entity_precision == 1.0
    assert m.entity_recall == 1.0
    assert m.entity_f1 == 1.0


def test_entity_precision_and_recall_differ_when_prediction_over_generates():
    m = agreement_metrics([_result("ai", ["A", "B", "C"], "positive")],
                          [_label("ai", ["A", "B"], "positive")])
    assert m.entity_precision == pytest.approx(2 / 3)
    assert m.entity_recall == 1.0


def test_entity_matching_is_case_insensitive():
    m = agreement_metrics([_result("ai", ["openai"], "positive")],
                          [_label("ai", ["OpenAI"], "positive")])
    assert m.entity_recall == 1.0


def test_unreviewed_labels_are_ignored_entirely():
    m = agreement_metrics([_result("ai", ["A"], "positive")],
                          [_label("ai", ["A"], "positive", reviewed=False)])
    assert m.entity_f1 is None
    assert m.unavailable_reason is not None
    assert m.labelled_coverage == 0


def test_no_labels_reports_unavailable_not_zero():
    m = agreement_metrics([_result("ai", ["A"], "positive")], [])
    assert m.stance_accuracy is None
    assert m.entity_f1 is None
    assert "no reviewed" in m.unavailable_reason.lower()


def test_stance_accuracy_counts_exact_matches():
    results = [_result("a", [], "positive"), _result("b", [], "negative")]
    labels = [_label("a", [], "positive"), _label("b", [], "neutral")]
    m = agreement_metrics(results, labels)
    assert m.stance_accuracy == 0.5


def test_kappa_is_zero_for_chance_agreement():
    # Both raters assign the same single class to everything: no information.
    assert cohens_kappa(["a", "a", "a"], ["a", "a", "a"]) is None


def test_kappa_is_one_for_perfect_agreement_across_classes():
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)


def test_kappa_is_negative_for_systematic_disagreement():
    k = cohens_kappa(["a", "b"], ["b", "a"])
    assert k is not None and k < 0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_agreement.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evals.agreement'`

- [ ] **Step 3: Write the implementation**

`api/evals/agreement.py`:

```python
"""Metrics measured against human-reviewed labels.

Stance is subjective, so it is reported with Cohen's kappa alongside raw
accuracy — kappa discounts the agreement you would get by chance, which raw
accuracy silently credits.

Kappa is hand-rolled rather than imported so the harness needs no scientific
stack for fifteen lines of arithmetic.
"""

from collections import Counter
from dataclasses import dataclass

from evals.golden import GoldenLabel, reviewed_only
from evals.metrics import ExtractionResult


@dataclass
class AgreementMetrics:
    labelled_coverage: int
    total_labels: int
    entity_precision: float | None
    entity_recall: float | None
    entity_f1: float | None
    stance_accuracy: float | None
    stance_kappa: float | None
    unavailable_reason: str | None


def cohens_kappa(rater_a: list[str], rater_b: list[str]) -> float | None:
    """Cohen's kappa. ``None`` when it is undefined.

    Undefined happens when expected agreement is 1.0 — every item in one class
    — where the statistic divides by zero. That is a real "cannot say", not a
    zero.
    """
    if not rater_a or len(rater_a) != len(rater_b):
        return None

    n = len(rater_a)
    observed = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n

    count_a = Counter(rater_a)
    count_b = Counter(rater_b)
    expected = sum(
        (count_a[label] / n) * (count_b[label] / n)
        for label in set(count_a) | set(count_b)
    )

    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def agreement_metrics(
    results: list[ExtractionResult], labels: list[GoldenLabel]
) -> AgreementMetrics:
    """Score predictions against human-reviewed labels only."""
    reviewed = reviewed_only(labels)
    by_topic = {label.topic: label for label in reviewed}

    paired = [
        (r, by_topic[r.topic])
        for r in results
        if not r.failed and r.analysis is not None and r.topic in by_topic
    ]

    if not paired:
        return AgreementMetrics(
            labelled_coverage=0,
            total_labels=len(labels),
            entity_precision=None,
            entity_recall=None,
            entity_f1=None,
            stance_accuracy=None,
            stance_kappa=None,
            unavailable_reason=(
                f"no reviewed labels matched the evaluated topics "
                f"({len(reviewed)}/{len(labels)} labels reviewed)"
            ),
        )

    true_positives = 0
    predicted_total = 0
    actual_total = 0
    predicted_stances: list[str] = []
    actual_stances: list[str] = []

    for result, label in paired:
        analysis = result.analysis
        assert analysis is not None  # narrowed above
        predicted = {e.name.casefold() for e in analysis.entities}
        actual = {name.casefold() for name in label.entities}
        true_positives += len(predicted & actual)
        predicted_total += len(predicted)
        actual_total += len(actual)
        predicted_stances.append(analysis.stance)
        actual_stances.append(label.stance)

    precision = true_positives / predicted_total if predicted_total else None
    recall = true_positives / actual_total if actual_total else None
    if precision and recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    accuracy = sum(
        1 for p, a in zip(predicted_stances, actual_stances, strict=True) if p == a
    ) / len(paired)

    return AgreementMetrics(
        labelled_coverage=len(paired),
        total_labels=len(labels),
        entity_precision=precision,
        entity_recall=recall,
        entity_f1=f1,
        stance_accuracy=accuracy,
        stance_kappa=cohens_kappa(predicted_stances, actual_stances),
        unavailable_reason=None,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_agreement.py -v
```

Expected: 9 passed

- [ ] **Step 5: Verify gates and commit**

```bash
cd api && uv run --python 3.12 --extra dev pytest -q \
  && uv run --python 3.12 --extra dev ruff check . \
  && uv run --python 3.12 --extra dev mypy newsninja evals
cd .. && git add api/evals/agreement.py api/tests/test_evals_agreement.py
git commit -m "Add label-backed agreement metrics with Cohen's kappa

Stance is subjective, so raw accuracy is reported alongside kappa, which
discounts chance agreement that accuracy silently credits. Kappa is
hand-rolled to avoid a scientific-stack dependency for fifteen lines of
arithmetic, and returns None where the statistic is genuinely undefined."
```

---

### Task 6: Rubric-scored LLM judge

**Files:**
- Create: `api/evals/judge.py`, `api/evals/prompts/judge.md`, `api/tests/test_evals_judge.py`

**Interfaces:**
- Consumes: `StructuredClient`, `ExtractionResult`
- Produces: `JudgeScore(coverage, neutrality, coherence, rationale)`, `JudgedMetrics(mean_coverage, mean_neutrality, mean_coherence, judged_count)`, `judge_summary(client, result) -> JudgeScore`, `judged_metrics(client, results) -> JudgedMetrics`, `JUDGE_MODEL`

- [ ] **Step 1: Write the failing test**

`api/tests/test_evals_judge.py`:

```python
import pytest

from evals.judge import JUDGE_MODEL, JudgeScore, judge_summary, judged_metrics
from evals.metrics import ExtractionResult
from newsninja.models import Article, ArticleAnalysis


def _result(topic="ai", summary="A summary."):
    return ExtractionResult(
        topic=topic,
        analysis=ArticleAnalysis(topic=topic, summary=summary, entities=[],
                                 stance="neutral", confidence=0.5, key_claims=[]),
        articles=[Article(title="t", url="u", source="google_news", body="body")],
        failed=False,
    )


class RecordingJudge:
    def __init__(self, score=None):
        self.score = score or {"coverage": 4, "neutrality": 5,
                               "coherence": 4, "rationale": "fine"}
        self.calls: list[dict] = []

    def structured(self, *, model, system, user, schema_model, max_retries=2):
        self.calls.append({"model": model, "user": user})
        return schema_model.model_validate(self.score)


def test_judge_returns_a_score():
    score = judge_summary(RecordingJudge(), _result())
    assert isinstance(score, JudgeScore)
    assert score.coverage == 4


def test_judge_uses_the_reasoning_model():
    client = RecordingJudge()
    judge_summary(client, _result())
    assert client.calls[0]["model"] == JUDGE_MODEL


def test_judge_is_shown_the_sources_not_just_the_summary():
    client = RecordingJudge()
    judge_summary(client, _result())
    assert "body" in client.calls[0]["user"], (
        "a judge that cannot see the sources cannot score coverage"
    )


def test_scores_are_bounded():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        JudgeScore(coverage=6, neutrality=3, coherence=3, rationale="r")


def test_judged_metrics_average_across_results():
    client = RecordingJudge()
    m = judged_metrics(client, [_result("a"), _result("b")])
    assert m.judged_count == 2
    assert m.mean_coverage == 4.0


def test_failed_extractions_are_not_judged():
    client = RecordingJudge()
    bad = ExtractionResult(topic="b", analysis=None, articles=[], failed=True)
    m = judged_metrics(client, [_result("a"), bad])
    assert m.judged_count == 1
    assert len(client.calls) == 1


def test_no_judgeable_results_reports_unavailable():
    m = judged_metrics(RecordingJudge(), [])
    assert m.judged_count == 0
    assert m.mean_coverage is None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_judge.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evals.judge'`

- [ ] **Step 3: Write the prompt**

```bash
mkdir -p api/evals/prompts
```

`api/evals/prompts/judge.md`:

```markdown
You are evaluating a news summary against the articles it was drawn from.

Score three axes, each 1 to 5:

- **coverage** — how much of what the articles collectively report is reflected in
  the summary. 5 means nothing important was dropped; 1 means the summary misses
  the main story.
- **neutrality** — how free the summary is of editorialising, loaded language, and
  positions the articles do not take. 5 means purely descriptive; 1 means the
  summary argues.
- **coherence** — how well the summary reads as connected prose. 5 means it flows
  as a paragraph; 1 means disjointed fragments.

Score only what is in front of you. Do not reward a summary for facts you happen
to know that the articles do not contain — that is coverage of the world, not
coverage of the sources.

Give a one-sentence rationale naming the single strongest reason for your scores.
```

- [ ] **Step 4: Write the implementation**

`api/evals/judge.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_judge.py -v
```

Expected: 7 passed

- [ ] **Step 6: Verify gates and commit**

```bash
cd api && uv run --python 3.12 --extra dev pytest -q \
  && uv run --python 3.12 --extra dev ruff check . \
  && uv run --python 3.12 --extra dev mypy newsninja evals
cd .. && git add api/evals/judge.py api/evals/prompts/judge.md api/tests/test_evals_judge.py
git commit -m "Add rubric-scored LLM judge for summary quality

Reported separately from the deterministic metrics because an LLM grading an
LLM is circular without calibration. The judge is shown the source articles,
not just the summary, so coverage means coverage of the sources rather than
of the world."
```

---

### Task 7: Runner, report, and documentation

**Files:**
- Create: `api/evals/report.py`, `api/evals/run.py`, `api/tests/test_evals_report.py`
- Modify: `api/README.md`, root `README.md`

**Interfaces:**
- Consumes: everything above
- Produces: `render_report(deterministic, agreement, judged, meta) -> str`, `main(argv=None) -> int`

- [ ] **Step 1: Write the failing test**

`api/tests/test_evals_report.py`:

```python
from evals.agreement import AgreementMetrics
from evals.judge import JudgedMetrics
from evals.metrics import DeterministicMetrics
from evals.report import render_report


def _det(**kw):
    base = dict(topics_evaluated=5, schema_valid_rate=1.0, grounding_rate=0.93,
                hallucinated_claim_rate=0.07, mean_claims_per_topic=3.2,
                mean_entities_per_topic=4.1)
    base.update(kw)
    return DeterministicMetrics(**base)


def _agree(**kw):
    base = dict(labelled_coverage=0, total_labels=0, entity_precision=None,
                entity_recall=None, entity_f1=None, stance_accuracy=None,
                stance_kappa=None, unavailable_reason="no reviewed labels")
    base.update(kw)
    return AgreementMetrics(**base)


def _meta():
    return {"model": "openai/gpt-oss-20b", "prompt_version": "1", "generated": "2026-07-31"}


def test_report_contains_the_deterministic_numbers():
    out = render_report(_det(), _agree(), JudgedMetrics(0, None, None, None), _meta())
    assert "0.93" in out or "93" in out
    assert "Deterministic" in out


def test_unavailable_metrics_say_so_rather_than_printing_zero():
    out = render_report(_det(), _agree(), JudgedMetrics(0, None, None, None), _meta())
    assert "unavailable" in out.lower()
    assert "no reviewed labels" in out
    assert "0.00" not in out.split("Agreement")[1].split("Judged")[0]


def test_report_records_provenance_metadata():
    out = render_report(_det(), _agree(), JudgedMetrics(0, None, None, None), _meta())
    assert "gpt-oss-20b" in out
    assert "prompt_version" in out.lower() or "prompt version" in out.lower()


def test_agreement_section_states_label_coverage():
    out = render_report(_det(), _agree(labelled_coverage=3, total_labels=5,
                                        entity_f1=0.8, entity_precision=0.75,
                                        entity_recall=0.86, stance_accuracy=0.66,
                                        stance_kappa=0.4, unavailable_reason=None),
                        JudgedMetrics(0, None, None, None), _meta())
    assert "3" in out and "5" in out


def test_judged_section_is_separated_from_deterministic():
    out = render_report(_det(), _agree(), JudgedMetrics(4, 4.2, 4.8, 4.0), _meta())
    assert out.index("Deterministic") < out.index("Judged")
    assert "4.2" in out
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_report.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evals.report'`

- [ ] **Step 3: Write the implementation**

`api/evals/report.py`:

```python
"""Markdown report generation.

The three metric families are kept visually separate because they carry
different epistemic weight: deterministic numbers are arithmetic, agreement
numbers depend on how many labels a human actually reviewed, and judged numbers
are one model's opinion of another's output.
"""

from typing import Any

from evals.agreement import AgreementMetrics
from evals.judge import JudgedMetrics
from evals.metrics import DeterministicMetrics


def _fmt(value: float | None, digits: int = 2) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def render_report(
    deterministic: DeterministicMetrics,
    agreement: AgreementMetrics,
    judged: JudgedMetrics,
    meta: dict[str, Any],
) -> str:
    lines: list[str] = [
        "# Evaluation report",
        "",
        f"Generated: {meta.get('generated', 'unknown')}  ",
        f"Extraction model: `{meta.get('model', 'unknown')}`  ",
        f"Prompt version: `{meta.get('prompt_version', 'unknown')}`  ",
        f"Topics evaluated: {deterministic.topics_evaluated}",
        "",
        "## Deterministic",
        "",
        "No model judges these. They are arithmetic over the extraction output "
        "and its source articles.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Schema validity rate | {_fmt(deterministic.schema_valid_rate)} |",
        f"| Quote grounding rate | {_fmt(deterministic.grounding_rate)} |",
        f"| Hallucinated claim rate | {_fmt(deterministic.hallucinated_claim_rate)} |",
        f"| Mean claims per topic | {_fmt(deterministic.mean_claims_per_topic, 1)} |",
        f"| Mean entities per topic | {_fmt(deterministic.mean_entities_per_topic, 1)} |",
        "",
        "## Agreement",
        "",
        f"Backed by {agreement.labelled_coverage} human-reviewed labels "
        f"out of {agreement.total_labels} in the golden set.",
        "",
    ]

    if agreement.unavailable_reason:
        lines += [
            f"**unavailable** — {agreement.unavailable_reason}.",
            "",
            "Correct drafted labels in `evals/data/golden.jsonl` and set "
            "`reviewed: true` to populate this section.",
            "",
        ]
    else:
        lines += [
            "| Metric | Value |",
            "| --- | --- |",
            f"| Entity precision | {_fmt(agreement.entity_precision)} |",
            f"| Entity recall | {_fmt(agreement.entity_recall)} |",
            f"| Entity F1 | {_fmt(agreement.entity_f1)} |",
            f"| Stance accuracy | {_fmt(agreement.stance_accuracy)} |",
            f"| Stance Cohen's kappa | {_fmt(agreement.stance_kappa)} |",
            "",
            "Kappa discounts chance agreement, which raw accuracy credits.",
            "",
        ]

    lines += [
        "## Judged",
        "",
        "One model's rubric score of another model's output, on a 1-5 scale. "
        "Reported separately because it is not ground truth.",
        "",
        f"Summaries judged: {judged.judged_count}",
        "",
        "| Axis | Mean |",
        "| --- | --- |",
        f"| Coverage | {_fmt(judged.mean_coverage, 1)} |",
        f"| Neutrality | {_fmt(judged.mean_neutrality, 1)} |",
        f"| Coherence | {_fmt(judged.mean_coherence, 1)} |",
        "",
    ]

    return "\n".join(lines)
```

`api/evals/run.py`:

```python
"""Run the evaluation harness and write a report.

Human-invoked; not part of CI. A full run makes real API calls against the
free tier, so it is rate-limited by the same limiter production uses.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from evals.agreement import agreement_metrics
from evals.corpus import load_corpus
from evals.golden import load_golden
from evals.judge import judged_metrics
from evals.metrics import ExtractionResult, deterministic_metrics
from evals.report import render_report

REPORT_DIR = Path(__file__).resolve().parents[2] / "docs" / "evals"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evals.run", description="Run the evaluation harness."
    )
    parser.add_argument("--report", action="store_true", help="Write the markdown report.")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge.")
    args = parser.parse_args(argv)

    from newsninja.analysis.client import GroqClient
    from newsninja.analysis.extract import DEFAULT_MODEL, extract_topic
    from newsninja.analysis.prompts import PROMPT_VERSION
    from newsninja.config import Settings
    from newsninja.errors import ExtractionFailure

    settings = Settings()
    client = GroqClient(api_key=settings.groq_api_key)
    records = load_corpus()

    results: list[ExtractionResult] = []
    for record in records:
        try:
            analysis = extract_topic(client, record.topic, record.articles)
            results.append(ExtractionResult(topic=record.topic, analysis=analysis,
                                            articles=record.articles, failed=False))
        except ExtractionFailure as exc:
            print(f"extraction failed for {record.topic!r}: {exc}", file=sys.stderr)
            results.append(ExtractionResult(topic=record.topic, analysis=None,
                                            articles=record.articles, failed=True))

    deterministic = deterministic_metrics(results)
    agreement = agreement_metrics(results, load_golden())
    judged = judged_metrics(client, results) if not args.no_judge else judged_metrics(
        client, []
    )

    report = render_report(
        deterministic, agreement, judged,
        {
            "generated": datetime.now(UTC).date().isoformat(),
            "model": DEFAULT_MODEL,
            "prompt_version": PROMPT_VERSION,
        },
    )
    print(report)

    if args.report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / "latest.md"
        path.write_text(report, encoding="utf-8")
        print(f"\nwritten to {path}", file=sys.stderr)

    print(
        f"\ntokens: {client.usage.prompt_tokens} prompt / "
        f"{client.usage.completion_tokens} completion / {client.usage.calls} calls",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd api && uv run --python 3.12 --extra dev pytest tests/test_evals_report.py -v
```

Expected: 5 passed

- [ ] **Step 5: Verify all three gates**

```bash
cd api && uv run --python 3.12 --extra dev pytest -q \
  && uv run --python 3.12 --extra dev ruff check . \
  && uv run --python 3.12 --extra dev mypy newsninja evals
```

- [ ] **Step 6: Update both READMEs**

In `api/README.md`, add an `## Evaluation` section describing: the three metric families and why they are separated; that the golden set is model-drafted and human-corrected, with only `reviewed: true` labels counting; and the commands (`python -m evals.capture`, `python -m evals.bootstrap`, `python -m evals.run --report`).

In the root `README.md`, replace the "Not yet built: the evaluation harness…" sentence under **Project status** with an accurate statement of what now exists. **Do not cite any metric value** unless `docs/evals/latest.md` exists and contains it — if no run has happened yet, say the harness exists and has not yet been run against a reviewed golden set.

- [ ] **Step 7: Commit**

```bash
cd .. && git add api/evals api/tests/test_evals_report.py api/README.md README.md
git commit -m "Add eval runner, markdown report, and documentation

The three metric families are reported separately because they carry
different epistemic weight. Metrics that cannot be computed print
'unavailable' with a reason rather than a zero that would read as a result."
```

---

## Self-Review

**Spec coverage (§6).** Golden set → Tasks 1, 2, 4. Deterministic metrics (schema validity, entity grounding, hallucination rate) → Task 3. Agreement metrics with Cohen's κ → Task 5. Rubric-scored judge reported separately → Task 6. `python -m evals.run --report` writing to `docs/evals/` → Task 7.

**Deliberate deviation from the spec.** §6 specifies ~40 hand-labelled articles authored by the developer. This plan drafts candidates with `gpt-oss-120b` for human correction instead, because a golden set that never gets written measures nothing. The honesty property is preserved by `reviewed: false` gating — drafted labels are invisible to every metric — and provenance is stated in the report and both READMEs. If the developer prefers to author labels from scratch, Task 4's bootstrapper is simply unused; nothing else changes.

**Spec item deferred:** §6 mentions calibrating the judge against a human-scored subset and reporting the correlation. The harness reports judged scores and label coverage separately, but the correlation figure needs human scores that do not exist yet. It is not implemented; the report does not claim it.

**Type consistency checked.** `ExtractionResult` is defined once in `metrics.py` and imported by `agreement.py`, `judge.py`, and `run.py`. `StructuredClient` (from Plan 1's `client.py`) types every client parameter. `GoldenLabel.stance` and `ArticleAnalysis.stance` share the same three-value literal. `source_corpus` and `ungrounded_claims` are used as Plan 1 defined them; nothing in `newsninja/` is modified.

**Known sharp edge:** Task 7's `run.py` extracts every corpus topic in one process. At five topics that is well inside the free-tier ceiling, but a much larger corpus would queue on the limiter. That is correct behaviour rather than a defect, and the runner prints its token usage so the cost is visible.
