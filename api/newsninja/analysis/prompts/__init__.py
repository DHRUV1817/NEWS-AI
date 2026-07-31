"""Versioned prompts. PROMPT_VERSION participates in the cache key, so editing
a prompt invalidates exactly the entries it affects."""

from pathlib import Path

PROMPT_VERSION = "1"

_DIR = Path(__file__).parent


def load(name: str) -> str:
    return (_DIR / f"{name}.md").read_text(encoding="utf-8")


EXTRACT_SYSTEM = load("extract")
