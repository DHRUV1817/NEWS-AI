#!/usr/bin/env python3
"""Fail when the documentation states a test count the suite does not produce.

A test count in a README is a claim, and this project's organising principle is
that every claim it makes must be mechanically checkable. Left unchecked the
number drifts on the first change that adds a test — it already had, twice, in
opposite directions across three files.

Reads counts from the two suites and compares them against every number the
documentation states. Prints what it found either way, so a failure says which
file to edit rather than only that something is wrong.

Usage:
    python scripts/check-stated-test-count.py <python_count> <site_count>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each entry is a file and a pattern whose single group is the stated count.
# Adding a claim elsewhere means adding it here; a claim this script cannot see
# is a claim nothing checks.
CLAIMS: list[tuple[str, str, str]] = [
    ("README.md", r"^(\d+) tests, all offline", "python"),
    (
        "docs/HANDOFF.md",
        r"\*\*Tests: (\d+) Python \+ (\d+) site\*\*",
        "both",
    ),
]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    actual = {"python": int(argv[1]), "site": int(argv[2])}
    print(f"suites report: {actual['python']} python, {actual['site']} site")

    problems: list[str] = []

    for filename, pattern, kind in CLAIMS:
        path = ROOT / filename
        text = path.read_text(encoding="utf-8")
        match = re.search(pattern, text, re.MULTILINE)

        if match is None:
            # A claim that vanished is not silently fine: either the wording
            # changed and this script needs updating, or the number was dropped
            # and nobody noticed.
            problems.append(
                f"{filename}: no test-count claim matched {pattern!r}. "
                f"If the wording changed, update CLAIMS in this script."
            )
            continue

        if kind == "python":
            stated = int(match.group(1))
            print(f"{filename}: states {stated} python")
            if stated != actual["python"]:
                problems.append(
                    f"{filename} states {stated} python tests, suite reports "
                    f"{actual['python']}"
                )
        else:
            stated_py, stated_site = int(match.group(1)), int(match.group(2))
            print(f"{filename}: states {stated_py} python, {stated_site} site")
            if stated_py != actual["python"]:
                problems.append(
                    f"{filename} states {stated_py} python tests, suite reports "
                    f"{actual['python']}"
                )
            if stated_site != actual["site"]:
                problems.append(
                    f"{filename} states {stated_site} site tests, suite reports "
                    f"{actual['site']}"
                )

    if problems:
        print("\nstated counts do not match the suites:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print("every stated count matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
