"""Per-IP request window.

This bounds how fast one client may call the service. It does not protect the
shared token budget — ten callers from ten addresses each pass this check and
still exhaust it. ``TokenBudgetLimiter``'s wait ceiling is what catches that.
"""

import time
from collections import deque
from collections.abc import Callable

WINDOW_SECONDS = 60.0


class RateLimiter:
    """Sliding window of request timestamps, keyed by client address."""

    def __init__(self, limit: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._limit = limit
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._last_eviction_time = -WINDOW_SECONDS  # Allow eviction on first call

    def tracked_clients(self) -> int:
        """How many addresses currently hold state. Reporting and tests only."""
        return len(self._hits)

    def _evict_idle(self, now: float) -> None:
        """Drop addresses with nothing left inside the window.

        Without this the key map grows once per distinct address seen and never
        shrinks, so a caller cycling source addresses could exhaust memory
        through the very component meant to bound abuse.

        This runs at most once per WINDOW_SECONDS to keep the operation O(1)
        amortised per request rather than O(n) in the number of clients.
        """
        if now - self._last_eviction_time < WINDOW_SECONDS:
            return
        self._last_eviction_time = now
        for key in list(self._hits.keys()):
            hits = self._hits[key]
            while hits and now - hits[0] >= WINDOW_SECONDS:
                hits.popleft()
            if not hits:
                del self._hits[key]

    def check(self, key: str) -> float | None:
        """Record a request for ``key``.

        Returns ``None`` when the request is allowed, otherwise the seconds
        until a slot frees. A refused request is not recorded — counting it
        would extend the block every time a blocked client retried.
        """
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= WINDOW_SECONDS:
            hits.popleft()

        if len(hits) >= self._limit:
            return hits[0] + WINDOW_SECONDS - now

        hits.append(now)
        self._evict_idle(now)
        return None
