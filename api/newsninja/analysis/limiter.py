"""Sliding-window token budget limiter for Groq's free tier.

Waits before a call that would breach the budget instead of retrying into a 429.
The provider's own ``x-ratelimit-remaining-tokens`` header feeds back in via
``observe`` so the limiter corrects for usage it did not account for.
"""

import time
from collections import deque
from collections.abc import Callable

WINDOW_SECONDS = 60.0


class TokenBudgetLimiter:
    def __init__(
        self,
        tpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._tpm = tpm
        self._clock = clock
        self._sleep = sleeper
        self._events: deque[tuple[float, int]] = deque()
        self._forced_wait_until: float = 0.0

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
            self._events.popleft()

    def _used(self, now: float) -> int:
        self._prune(now)
        return sum(tokens for _, tokens in self._events)

    def reserve(self, estimated_tokens: int) -> None:
        """Block until ``estimated_tokens`` fits inside the budget, then record it."""
        if estimated_tokens > self._tpm:
            raise ValueError(
                f"reservation of {estimated_tokens} exceeds the entire "
                f"per-minute budget of {self._tpm}; split the request"
            )

        now = self._clock()
        if now < self._forced_wait_until:
            self._sleep(self._forced_wait_until - now)
            now = self._clock()

        while self._used(now) + estimated_tokens > self._tpm:
            oldest_at, _ = self._events[0]
            self._sleep(max(0.0, oldest_at + WINDOW_SECONDS - now))
            now = self._clock()

        self._events.append((now, estimated_tokens))

    def observe(self, remaining: int, reset_seconds: float) -> None:
        """Record the provider's own view of the budget.

        When the provider says very little is left, force a wait until its stated
        reset regardless of what the local window believes.
        """
        if remaining < self._tpm * 0.05:
            self._forced_wait_until = self._clock() + reset_seconds
