# newsninja (core package)

Turns a list of topics into a source-grounded news briefing with audio, on
Groq's free tier.

## Requirements

Python 3.12. The system Python on macOS is 3.9 and will not work — use `uv`.

## Setup

    cp ../.env.example ../.env    # then set GROQ_API_KEY
    uv run --python 3.12 --extra dev pytest

## Usage

    uv run --python 3.12 newsninja --topic "artificial intelligence" --topic "climate"

Exit codes: `0` success, `1` missing or invalid configuration, `2` the model
never returned schema-valid output, `3` the provider's rate limit was hit — the
message says roughly how long to wait.

## Design notes

**Structured output.** Extraction uses Groq's strict `json_schema` mode, which is
supported only on `openai/gpt-oss-20b` and `openai/gpt-oss-120b`; `GroqClient`
refuses any other model rather than quietly falling back to free-form text.
Pydantic's schema output needs a transform first — see `analysis/schema.py`. A
response that fails validation is retried with the validation error fed back to
the model, and raises `ExtractionFailure` once the retry budget is spent.

**Grounding.** The extraction prompt requires every `Claim` to carry a `quote`
copied character-for-character from the supplied article text.
`analysis/grounding.py` is what checks it: `grounding_rate` reports the fraction
of quotes appearing verbatim in the concatenated sources, by exact substring
containment — no normalisation, no fuzzy matching, no model in the loop. That is
the sense in which hallucination here is a string containment check rather than
a judgement.

What it does not mean: nothing rejects an ungrounded claim at runtime. The check
is a measurement utility, deliberately outside the pipeline's control flow, and
is what the eval harness will read.

**Rate limits.** The free tier is tokens-per-minute limited, per model, so work is
spread across models — extraction on `gpt-oss-20b`, synthesis on `gpt-oss-120b`,
translation on `llama-3.3-70b-versatile` — and articles are batched one call per
topic rather than one per article. `analysis/limiter.py` holds a sliding
sixty-second window per model: a call reserves an estimate that includes the
completion tokens it expects to spend, then settles that reservation against the
usage the provider actually reported, and the provider's own
`x-ratelimit-remaining-tokens` header forces a wait when it disagrees with the
local view. A model with no configured budget raises rather than skipping the
limiter.

**Audio.** gTTS by default, across twelve languages, returning MP3. Orpheus
(`ENABLE_ORPHEUS=true`) is an opt-in upgrade covering English and Saudi Arabic
only; it POSTs to Groq's `/openai/v1/audio/speech` endpoint and returns WAV.
`--out` is written verbatim, so name the file for whichever backend you expect.

Orpheus returns HTTP 400 `model_terms_required` until its terms are accepted at
console.groq.com, so on a fresh account every Orpheus call fails and gTTS — the
fallback — is the path that actually runs. Long scripts are chunked, and no
chunk exceeds the limit even when the text carries no punctuation the splitter
recognises.

**Sources.** Google News RSS needs no authentication. Reddit requires a free
script app at reddit.com/prefs/apps — the public JSON endpoint returns 403 — and
without credentials the source reports itself unavailable, which the CLI prints
as a skipped source, distinct from a failure. A source that is tried and fails
has every failure recorded against its name and the run continues on what is
left.

## Evaluation

`api/evals/` measures the extraction pipeline, split into three families that
are reported separately because they carry different epistemic weight:

- **Deterministic** (`evals/metrics.py`) — schema validity, quote-grounding
  rate, and ungrounded-claim rate. Pure arithmetic over the model's output and
  its source articles; no model judges any of it, and a metric that cannot be
  computed is `None`, never `0.0`.
- **Agreement** (`evals/agreement.py`) — entity precision/recall/F1 and stance
  accuracy plus Cohen's kappa, scored against the golden set. Kappa is
  reported alongside accuracy because it discounts the agreement you would get
  by chance.
- **Judged** (`evals/judge.py`) — a rubric score (coverage, neutrality,
  coherence, 1-5) from one Groq model grading another model's summary. This is
  the least objective family and is presented as one model's opinion, not
  ground truth.

Deterministic and agreement count entities differently, on purpose. Agreement
case-folds
each topic's entities into a set before scoring, so a topic naming "Apple"
four times contributes one entity and repetition cannot buy precision.
`mean_entities_per_topic` counts every emission, so the same topic contributes
four — it describes how much the model said, not how much of it was distinct.
Expect the two numbers to disagree on identical data; neither is a typo for
the other.

The golden set (`evals/data/golden.jsonl`) is meant to be model-drafted and
human-corrected: `python -m evals.bootstrap` drafts candidate labels with
`gpt-oss-120b` for a human to review and correct, but every drafted label
starts `reviewed: false` and is invisible to the agreement metrics until a
human sets `reviewed: true`. Presenting an unreviewed, model-drafted label as
ground truth would make the agreement numbers circular.

No golden set has been created yet and no run has happened, so this README
cites no metric value from any of the three families — there is nothing
measured yet to cite. Until `evals/data/golden.jsonl` exists with reviewed
labels, the agreement section of a generated report reads `unavailable`.

Commands, run from `api/`:

```bash
python -m evals.capture           # fetch and commit a stable article corpus
python -m evals.bootstrap         # draft golden labels for human correction
python -m evals.run --report      # run all three metric families, write docs/evals/latest.md
```

All three are human-invoked and not part of CI; `evals.run` makes real Groq
API calls against the free tier and is rate-limited by the same limiter
production uses.

## Not here yet

No web UI. `api/evals/` (above) covers extraction quality; there is no
evaluation of synthesis, translation, or audio output yet. No test in this
package makes a network call.
