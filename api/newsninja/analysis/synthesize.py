"""Briefing synthesis and translation.

Both produce prose rather than structured data, so they use free-form calls and
can run on llama-3.3-70b-versatile, which carries the highest free-tier token
budget (12,000 TPM). Synthesis uses gpt-oss-120b for its stronger reasoning.
"""

from newsninja.analysis.client import TextClient
from newsninja.analysis.prompts import SYNTHESIZE_SYSTEM, TRANSLATE_SYSTEM
from newsninja.models import ArticleAnalysis, Briefing

SYNTHESIS_MODEL = "openai/gpt-oss-120b"
TRANSLATION_MODEL = "llama-3.3-70b-versatile"


def _render(analyses: list[ArticleAnalysis]) -> str:
    blocks = []
    for analysis in analyses:
        entities = ", ".join(entity.name for entity in analysis.entities) or "none"
        claims = "\n".join(f"  - {claim.text}" for claim in analysis.key_claims) or "  - none"
        blocks.append(
            f"Topic: {analysis.topic}\n"
            f"Summary: {analysis.summary}\n"
            f"Stance: {analysis.stance} (confidence {analysis.confidence:.2f})\n"
            f"Entities: {entities}\n"
            f"Key claims:\n{claims}"
        )
    return "\n\n".join(blocks)


def synthesize(
    client: TextClient, analyses: list[ArticleAnalysis], model: str = SYNTHESIS_MODEL
) -> str:
    """Turn per-topic analyses into a spoken-word briefing script."""
    return client.text(model=model, system=SYNTHESIZE_SYSTEM, user=_render(analyses))


def translate(
    client: TextClient, script: str, language: str, model: str = TRANSLATION_MODEL
) -> str:
    """Translate ``script`` into ``language``. English is a no-op."""
    if language == "en":
        return script
    return client.text(
        model=model,
        system=TRANSLATE_SYSTEM,
        user=f"Target language code: {language}\n\nScript:\n{script}",
    )


def build_briefing(
    client: TextClient, analyses: list[ArticleAnalysis], language: str = "en"
) -> Briefing:
    """Synthesise, translate if needed, and package the result."""
    script = synthesize(client, analyses)
    script = translate(client, script, language)
    return Briefing(
        topics=[analysis.topic for analysis in analyses],
        script=script,
        analyses=analyses,
        language=language,
    )
