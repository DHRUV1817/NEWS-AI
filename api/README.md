# newsninja (core package)

Turns a list of topics into a source-grounded news briefing with audio.

## Requirements

Python 3.12. The system Python on macOS is 3.9 and will not work — use `uv`.

## Setup

    cp ../.env.example ../.env    # then set GROQ_API_KEY
    uv run --python 3.12 --extra dev pytest

## Usage

    uv run --python 3.12 newsninja --topic "artificial intelligence" --topic "climate"

## Design notes

**Structured output.** Extraction uses Groq's strict `json_schema` mode, which is
supported only on `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. Pydantic's schema
output needs a transform first — see `analysis/schema.py`.

**Grounding.** Every `Claim` carries a `quote` that must appear verbatim in its source
article, which makes hallucination a string containment check rather than a judgement.

**Rate limits.** The free tier is tokens-per-minute limited. Work is spread across models
(limits are per model) and articles are batched one call per topic rather than one per
article. See `analysis/limiter.py`.

**Audio.** gTTS by default across twelve languages. Orpheus is an opt-in upgrade covering
English and Saudi Arabic only, and requires accepting model terms at console.groq.com.
Failures fall back to gTTS.

**Reddit.** Requires a free script app at reddit.com/prefs/apps. Without credentials the
source reports itself unavailable; the public JSON endpoint returns 403.
