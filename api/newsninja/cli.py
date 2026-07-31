"""Command-line entry point.

Makes the whole pipeline runnable without a web server, which is what lets the
eval harness and CI exercise it directly.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newsninja", description="Generate a source-grounded news briefing."
    )
    parser.add_argument("--topic", action="append", required=True,
                        help="Topic to analyse. Repeat for up to 5.")
    parser.add_argument("--language", default="en", help="Output language code.")
    parser.add_argument("--out", type=Path, default=Path("briefing.mp3"),
                        help="Where to write the audio.")
    parser.add_argument("--no-audio", action="store_true",
                        help="Print the script only; skip speech synthesis.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the cache.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from newsninja.config import Settings

    try:
        settings = Settings()
    except ValidationError:
        print(
            "GROQ_API_KEY is not set. Add it to .env or export it, then retry.",
            file=sys.stderr,
        )
        return 1

    from newsninja.analysis.client import GroqClient
    from newsninja.cache import Cache
    from newsninja.pipeline import run_pipeline
    from newsninja.sources.google_news import GoogleNewsSource
    from newsninja.sources.reddit import RedditSource

    sources = [
        GoogleNewsSource(timeout=settings.request_timeout),
        RedditSource(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        ),
    ]

    pipeline_kwargs: dict[str, Any] = {
        "topics": args.topic,
        "sources": sources,
        "client": GroqClient(api_key=settings.groq_api_key),
        "cache": None if args.no_cache else Cache(settings.cache_path),
        "language": args.language,
        "enable_orpheus": settings.enable_orpheus,
        # Orpheus is a separate REST endpoint, so the key has to reach the
        # speech seam as well as the chat client.
        "api_key": settings.groq_api_key,
    }
    if args.no_audio:
        pipeline_kwargs["tts"] = lambda text, lang, orpheus: b""

    result = run_pipeline(**pipeline_kwargs)

    print(result.briefing.script)

    # Two different things, reported differently: a skipped source is a missing
    # capability the user can fix by adding credentials, a failed source is a
    # source that was tried and broke.
    for name in result.skipped_sources:
        print(
            f"note: source {name} was skipped (unavailable — check its credentials)",
            file=sys.stderr,
        )

    for name, messages in result.source_errors.items():
        for message in messages:
            print(f"warning: source {name} failed: {message}", file=sys.stderr)

    if not args.no_audio:
        args.out.write_bytes(result.audio)
        print(f"\nAudio written to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
