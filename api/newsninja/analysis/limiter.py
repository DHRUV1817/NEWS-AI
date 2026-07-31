"""Sliding-window token budget limiter for Groq's free tier.

Waits before a call that would breach the budget instead of retrying into a 429.

Three inputs keep the window honest. ``reserve`` books an estimate *before* the
call, including the completion tokens the response is expected to cost — a
prompt-only reservation under-counts by roughly 3x and sails straight through
the cap. ``settle`` replaces that estimate with the usage the provider actually
reported. ``observe`` feeds the provider's own ``x-ratelimit-remaining-tokens``
back in, so usage this process never saw still slows it down.
"""

import time
from collections import deque
from collections.abc import Callable

WINDOW_SECONDS = 60.0

# Below this fraction of the budget, trust the provider's reset over the local
# window and wait it out.
LOW_REMAINING_FRACTION = 0.05


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
        # (booked_at, tokens, reservation_id)
        self._events: deque[tuple[float, int, int]] = deque()
        self._next_reservation = 0
        self._forced_wait_until: float = 0.0

    @property
    def tpm(self) -> int:
        return self._tpm

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
            self._events.popleft()

    def _used(self, now: float) -> int:
        self._prune(now)
        return sum(tokens for _, tokens, _ in self._events)

    def used_tokens(self) -> int:
        """Tokens booked in the current window. Reporting only."""
        return self._used(self._clock())

    def reserve(self, estimated_tokens: int) -> int:
        """Block until ``estimated_tokens`` fits inside the budget, then book it.

        Returns a reservation id to hand to ``settle`` once the real cost of the
        call is known.
        """
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
            oldest_at, _, _ = self._events[0]
            self._sleep(max(0.0, oldest_at + WINDOW_SECONDS - now))
            now = self._clock()

        reservation = self._next_reservation
        self._next_reservation += 1
        self._events.append((now, estimated_tokens, reservation))
        return reservation

    def settle(self, reservation: int, actual_tokens: int) -> None:
        """Replace a reservation's estimate with the call's real total usage.

        A reservation that has already aged out of the window is left alone: it
        is more than sixty seconds old and no longer counts against the budget.
        """
        for index, (booked_at, _estimate, identifier) in enumerate(self._events):
            if identifier == reservation:
                self._events[index] = (booked_at, actual_tokens, identifier)
                return

    def observe(self, remaining: int, reset_seconds: float) -> None:
        """Record the provider's own view of the budget.

        When the provider says very little is left, force a wait until its stated
        reset regardless of what the local window believes.
        """
        if remaining < self._tpm * LOW_REMAINING_FRACTION:
            self._forced_wait_until = self._clock() + reset_seconds
