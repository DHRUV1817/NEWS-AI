"""Sliding-window token budget limiter for Groq's free tier.

Waits before a call that would breach the budget instead of retrying into a 429.

Three inputs keep the window honest. ``reserve`` books an estimate *before* the
call, including the completion tokens the response is expected to cost — a
prompt-only reservation under-counts by roughly 3x and sails straight through
the cap. ``settle`` replaces that estimate with the usage the provider actually
reported. ``observe`` feeds the provider's own ``x-ratelimit-remaining-tokens``
back in, so usage this process never saw still slows it down.

One limiter is shared by concurrent callers — ``newsninja.api.deps.get_client``
is a process singleton and every HTTP handler is a plain ``def``, so FastAPI
runs them in a threadpool against one instance. Every method that reads or
writes the window therefore holds ``_lock``. ``reserve`` releases it around the
sleep: holding it while waiting would queue every other caller behind this one
rather than letting them see the window it is waiting on.
"""

import threading
import time
from collections import deque
from collections.abc import Callable

from newsninja.errors import RateLimitError

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
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
            self._events.popleft()

    def _used(self, now: float) -> int:
        self._prune(now)
        return sum(tokens for _, tokens, _ in self._events)

    def used_tokens(self) -> int:
        """Tokens booked in the current window. Reporting only."""
        with self._lock:
            return self._used(self._clock())

    def _wait_or_refuse(self, wait: float, deadline: float | None) -> None:
        """Sleep for ``wait``, unless the caller's deadline forbids it.

        The deadline is fixed once, at ``reserve`` entry, so what it bounds is
        the *total* time spent inside one call. Comparing each sleep against the
        ceiling separately bounded nothing: a reservation that has to retire six
        window events sleeps six times, and six waits of 2.0s are each under a
        5.0s ceiling while summing to 12.0s inside a single call.

        ``retry_after`` carries the wait that was declined rather than the time
        left on the deadline: the caller needs to know when the budget actually
        frees, not how long this particular caller was willing to hold on.
        """
        if deadline is not None:
            remaining = deadline - self._clock()
            if wait > remaining:
                raise RateLimitError(
                    f"the token budget needs {wait:.1f}s to clear, which exceeds "
                    f"the {max(0.0, remaining):.1f}s this caller has left of its "
                    f"wait ceiling",
                    retry_after=wait,
                )
        self._sleep(wait)

    def reserve(self, estimated_tokens: int, max_wait: float | None = None) -> int:
        """Block until ``estimated_tokens`` fits inside the budget, then book it.

        ``max_wait`` bounds the whole call, not each individual sleep: a
        deadline is fixed on entry and a wait that would carry the call past it
        raises ``RateLimitError`` instead of sleeping. Callers that can afford to
        wait — the CLI, the eval harness — pass nothing and behave as before.

        Returns a reservation id to hand to ``settle`` once the real cost of the
        call is known.
        """
        if estimated_tokens > self._tpm:
            raise ValueError(
                f"reservation of {estimated_tokens} exceeds the entire "
                f"per-minute budget of {self._tpm}; split the request"
            )

        deadline = None if max_wait is None else self._clock() + max_wait

        while True:
            with self._lock:
                now = self._clock()
                if now < self._forced_wait_until:
                    wait = self._forced_wait_until - now
                elif self._used(now) + estimated_tokens > self._tpm:
                    oldest_at, _, _ = self._events[0]
                    wait = max(0.0, oldest_at + WINDOW_SECONDS - now)
                else:
                    # Reading the counter, incrementing it and appending the
                    # event are one step: split apart, two threads take the
                    # same id and the second settle() corrects the wrong event.
                    reservation = self._next_reservation
                    self._next_reservation += 1
                    self._events.append((now, estimated_tokens, reservation))
                    return reservation

            # Outside the lock on purpose — see the module docstring — so the
            # window is re-read from scratch on the next pass.
            self._wait_or_refuse(wait, deadline)

    def settle(self, reservation: int, actual_tokens: int) -> None:
        """Replace a reservation's estimate with the call's real total usage.

        A reservation that has already aged out of the window is left alone: it
        is more than sixty seconds old and no longer counts against the budget.
        """
        with self._lock:
            for index, (booked_at, _estimate, identifier) in enumerate(self._events):
                if identifier == reservation:
                    self._events[index] = (booked_at, actual_tokens, identifier)
                    return

    def observe(self, remaining: int, reset_seconds: float) -> None:
        """Record the provider's own view of the budget.

        When the provider says very little is left, force a wait until its stated
        reset regardless of what the local window believes.
        """
        with self._lock:
            if remaining < self._tpm * LOW_REMAINING_FRACTION:
                self._forced_wait_until = self._clock() + reset_seconds
