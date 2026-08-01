#!/usr/bin/env python3
"""Fail when the documentation states a test count the suites do not produce.

A test count in a README is a claim, and this project's organising principle is
that every claim it makes must be mechanically checkable. Left unchecked the
number drifts on the first change that adds a test — it already had, twice, in
opposite directions across three files.

Counts are read from each runner's machine-readable report rather than scraped
from its human-facing summary. The first version of this script grepped that
summary and failed on a runner while passing locally, because the line it was
reading is formatted for people and formatting is allowed to change. The same
reasoning the project applies to its evaluation report applies here: read a
contract, not prose.

Usage:
    python scripts/check-stated-test-count.py <pytest-junit.xml> <vitest.json>
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each entry is a file, a pattern whose groups are the stated counts, and which
# suites those groups refer to. Adding a claim elsewhere means adding it here;
# a claim this script cannot see is a claim nothing checks.
CLAIMS: list[tuple[str, str, str]] = [
    ("README.md", r"^(\d+) tests, all offline", "python"),
    ("docs/HANDOFF.md", r"\*\*Tests: (\d+) Python \+ (\d+) site\*\*", "both"),
]


def pytest_passed(path: Path) -> int:
    """Passing tests from a JUnit XML report.

    ``tests`` counts everything attempted, so the ones that did not pass come
    back off it. A run with failures should not quietly satisfy a claim about
    how many tests pass.
    """
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise SystemExit(f"{path}: no testsuite element")
    attrs = suite.attrib
    total = int(attrs["tests"])
    not_passed = (
        int(attrs.get("errors", 0))
        + int(attrs.get("failures", 0))
        + int(attrs.get("skipped", 0))
    )
    return total - not_passed


def vitest_passed(path: Path) -> int:
    return int(json.loads(path.read_text(encoding="utf-8"))["numPassedTests"])


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    actual = {
        "python": pytest_passed(Path(argv[1])),
        "site": vitest_passed(Path(argv[2])),
    }
    print(f"suites report: {actual['python']} python, {actual['site']} site")

    problems: list[str] = []

    for filename, pattern, kind in CLAIMS:
        text = (ROOT / filename).read_text(encoding="utf-8")
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

        stated = [int(g) for g in match.groups()]
        expected = [actual["python"]] if kind == "python" else [
            actual["python"],
            actual["site"],
        ]
        labels = ["python"] if kind == "python" else ["python", "site"]

        print(
            f"{filename}: states "
            + ", ".join(f"{n} {label}" for n, label in zip(stated, labels))
        )

        for n, want, label in zip(stated, expected, labels):
            if n != want:
                problems.append(
                    f"{filename} states {n} {label} tests, suite reports {want}"
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
