"""SQLite response cache.

Keyed on the full call identity — topic, source, model, and prompt version —
so changing a prompt invalidates exactly the affected entries. Without this the
eval suite would exhaust the free tier on every CI run.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
)
"""


class Cache:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    @staticmethod
    def make_key(**parts: str) -> str:
        """Order-independent cache key over the full call identity."""
        canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM entries WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO entries (key, value) VALUES (?, ?)",
                (key, value),
            )

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM entries")
