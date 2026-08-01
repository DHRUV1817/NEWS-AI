# Evaluation report

Generated: 2026-08-01  
Extraction model: `openai/gpt-oss-20b`  
Prompt version: `1`  
Topics evaluated: 5  

## Deterministic

No model judges these. They are arithmetic over the extraction output and its source articles.

A quote is checked against each article's title and RSS summary text, not its full prose: the free Google News RSS source this harness reads from does not carry full article bodies. This limits what "grounded" can mean here — see `Claims counted` below for how much text that judgement rests on.

| Metric | Value |
| --- | --- |
| Schema validity rate | 1.00 |
| Claims counted | 12 |
| Quote grounding rate | 1.00 |
| Ungrounded claim rate | 0.00 |
| Mean claims per topic | 2.4 |
| Mean entities per topic | 13.4 |

## Agreement

Backed by 0 human-reviewed labels out of 5 in the golden set.

**unavailable** — no reviewed labels matched the evaluated topics (0/5 labels reviewed).

Correct drafted labels in `evals/data/golden.jsonl` and set `reviewed: true` to populate this section.

## Judged

One model's rubric score of another model's output, on a 1-5 scale. Reported separately because it is not ground truth.

Summaries judged: 0

| Axis | Mean |
| --- | --- |
| Coverage | unavailable |
| Neutrality | unavailable |
| Coherence | unavailable |
