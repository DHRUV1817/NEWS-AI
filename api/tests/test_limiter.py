import sys
import threading
from contextlib import contextmanager

import pytest

from newsninja.analysis.limiter import TokenBudgetLimiter
from newsninja.errors import RateLimitError


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _limiter(tpm: int = 1000) -> tuple[TokenBudgetLimiter, FakeClock]:
    clock = FakeClock()
    return TokenBudgetLimiter(tpm=tpm, clock=clock.time, sleeper=clock.sleep), clock


def test_reserve_under_budget_does_not_sleep():
    limiter, clock = _limiter()
    limiter.reserve(400)
    assert clock.slept == []


def test_reserve_over_budget_sleeps_until_window_clears():
    limiter, clock = _limiter(tpm=1000)
    limiter.reserve(800)
    limiter.reserve(400)  # 1200 > 1000, must wait for the window to roll
    assert clock.slept, "expected the limiter to wait"


def test_window_rolls_after_sixty_seconds():
    limiter, clock = _limiter(tpm=1000)
    limiter.reserve(900)
    clock.now += 61
    limiter.reserve(900)
    assert clock.slept == [], "old usage should have aged out of the window"


def test_observe_low_remaining_forces_a_wait():
    limiter, clock = _limiter(tpm=1000)
    limiter.observe(remaining=10, reset_seconds=5.0)
    limiter.reserve(500)
    assert clock.slept == [5.0]


def test_reservation_larger_than_the_whole_budget_raises():
    limiter, _ = _limiter(tpm=1000)
    with pytest.raises(ValueError):
        limiter.reserve(1001)


def test_settle_replaces_the_estimate_with_actual_usage():
    """Completion tokens used to be invisible: only the estimate was ever booked."""
    limiter, clock = _limiter(tpm=1000)
    reservation = limiter.reserve(100)
    limiter.settle(reservation, 900)

    assert limiter.used_tokens() == 900
    limiter.reserve(200)  # 900 + 200 > 1000
    assert clock.slept, "the window must reflect what the call really cost"


def test_settle_can_release_budget_when_a_call_came_in_cheap():
    limiter, clock = _limiter(tpm=1000)
    reservation = limiter.reserve(900)
    limiter.settle(reservation, 100)

    limiter.reserve(800)
    assert clock.slept == [], "an over-estimate must not hold the budget hostage"


def test_settle_ignores_a_reservation_that_aged_out_of_the_window():
    limiter, clock = _limiter(tpm=1000)
    reservation = limiter.reserve(100)
    clock.now += 61
    limiter.settle(reservation, 900)

    assert limiter.used_tokens() == 0


def test_reservations_are_distinct_per_call():
    limiter, _ = _limiter(tpm=1000)
    first = limiter.reserve(100)
    second = limiter.reserve(100)
    assert first != second

    limiter.settle(second, 500)
    assert limiter.used_tokens() == 600, "settling one call must not touch the other"


def test_bounded_reserve_refuses_when_the_wait_exceeds_the_ceiling():
    limiter, clock = _limiter(tpm=1000)
    limiter.reserve(800)
    with pytest.raises(RateLimitError):
        limiter.reserve(400, max_wait=5.0)
    assert clock.slept == [], "a refused reservation must not sleep at all"


def test_bounded_reserve_reports_the_declined_wait_as_retry_after():
    """retry_after must be the wait the limiter declined, not the ceiling.

    Reporting the ceiling would tell the caller to come back in 5s when the
    budget needs 60s, producing a retry storm against a budget already full.
    """
    limiter, clock = _limiter(tpm=1000)
    limiter.reserve(800)
    clock.now += 10.0
    with pytest.raises(RateLimitError) as caught:
        limiter.reserve(400, max_wait=5.0)
    assert caught.value.retry_after == pytest.approx(50.0)


def test_bounded_reserve_sleeps_when_the_wait_fits_the_ceiling():
    limiter, clock = _limiter(tpm=1000)
    limiter.reserve(800)
    clock.now += 55.0
    limiter.reserve(400, max_wait=30.0)
    assert clock.slept == [pytest.approx(5.0)]


def test_bounded_reserve_also_bounds_the_forced_wait_path():
    """observe() sets a forced wait on a different code path from the window loop.

    Bounding only the window loop lets a provider-signalled backoff hold the
    socket open anyway, which is the exact failure the ceiling exists to prevent.
    """
    limiter, clock = _limiter(tpm=1000)
    limiter.observe(remaining=10, reset_seconds=90.0)
    with pytest.raises(RateLimitError) as caught:
        limiter.reserve(100, max_wait=5.0)
    assert caught.value.retry_after == pytest.approx(90.0)
    assert clock.slept == []


def test_the_wait_ceiling_bounds_the_whole_call_not_each_sleep():
    """Verified failure: this slept [2.0] * 6 — 12.0s under a 5.0s ceiling.

    A reservation that has to retire several window events sleeps once per
    event. Checking each sleep against the ceiling separately bounds nothing:
    every one of those six waits was inside 5.0s. §10.1 of the spec promises at
    most `api_max_wait_seconds` of limiter wait per call, so the ceiling has to
    be a deadline fixed on entry.
    """
    limiter, clock = _limiter(tpm=8000)
    for _ in range(20):
        limiter.reserve(400)
        clock.now += 2.0
    # 8,000 booked at t=0,2,...,38. From t=58 the oldest expires in 2.0s and
    # each further event 2.0s after that, so freeing the 2,400 this reservation
    # needs means retiring six events: six separate 2.0s sleeps.
    clock.now = 58.0
    assert limiter.used_tokens() == 8000

    with pytest.raises(RateLimitError) as caught:
        limiter.reserve(2400, max_wait=5.0)

    assert sum(clock.slept) <= 5.0, (
        f"slept {clock.slept} — {sum(clock.slept)}s total under a 5.0s ceiling"
    )
    assert clock.slept == [pytest.approx(2.0), pytest.approx(2.0)]
    # The wait that was declined, not what was left of the ceiling: the caller
    # needs to know when the budget frees.
    assert caught.value.retry_after == pytest.approx(2.0)


def test_an_unbounded_reserve_still_waits_as_long_as_it_takes():
    """The CLI and the eval harness pass no ceiling and must not be refused."""
    limiter, clock = _limiter(tpm=8000)
    for _ in range(20):
        limiter.reserve(400)
        clock.now += 2.0
    clock.now = 58.0

    limiter.reserve(2400)
    assert sum(clock.slept) == pytest.approx(12.0)
    assert limiter.used_tokens() <= 8000


# --- concurrency ---
#
# One limiter is shared: get_client is a process singleton and every handler is
# a plain `def`, so FastAPI runs them in a threadpool against one instance.

#: Enough trials that the race shows up on every run rather than most runs.
#: Measured with the lock removed: over 5 runs of 40 trials each, the budget was
#: breached or a worker crashed on every run, but never on every trial.
TRIALS = 200


@contextmanager
def _a_widened_race_window():
    """Force the interpreter to switch threads far more often than usual.

    Without this a check-then-act sequence this short usually completes inside
    one scheduling slice, so an unsynchronised limiter passes by luck rather
    than by construction.
    """
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _reserve_from_threads(
    limiter: TokenBudgetLimiter, threads: int, tokens: int
) -> tuple[list[int], list[RuntimeError | IndexError]]:
    """Have ``threads`` reserve ``tokens`` each, as simultaneously as possible.

    Returns the reservation ids granted and anything raised that was not the
    expected refusal — an unsynchronised deque raises ``RuntimeError: deque
    mutated during iteration``, which would otherwise die inside a worker and
    leave the caller looking at a plausible-seeming total.
    """
    ready = threading.Barrier(threads)
    granted: list[int] = []
    crashes: list[RuntimeError | IndexError] = []
    record = threading.Lock()

    def _worker() -> None:
        ready.wait()
        try:
            reservation = limiter.reserve(tokens, max_wait=5.0)
        except RateLimitError:
            return
        except (RuntimeError, IndexError) as exc:
            # The two shapes an unsynchronised window takes: a deque mutated
            # mid-sum, and `self._events[0]` read after another thread emptied
            # it. Anything else propagates and pytest fails the test on it.
            with record:
                crashes.append(exc)
            return
        with record:
            granted.append(reservation)

    workers = [threading.Thread(target=_worker) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return granted, crashes


def test_concurrent_reserves_never_book_past_the_budget():
    """Verified failure: with the lock removed, trials booked 9,600 of 8,000.

    Twelve threads reserving 2,400 against an 8,000 TPM ceiling: three fit, the
    rest must be refused. Every thread reads the window and appends to it
    separately, so without synchronisation several pass the check before any of
    them books, and the window ends up over the ceiling it exists to hold.
    """
    worst = 0
    with _a_widened_race_window():
        for _ in range(TRIALS):
            limiter = TokenBudgetLimiter(tpm=8_000)
            _, crashes = _reserve_from_threads(limiter, threads=12, tokens=2_400)
            assert not crashes, f"a reserving thread crashed: {crashes[0]!r}"
            worst = max(worst, limiter.used_tokens())

    assert worst <= 8_000, (
        f"the window reached {worst} tokens against an 8,000 ceiling"
    )


def test_concurrent_reserves_get_distinct_reservation_ids():
    """`self._next_reservation += 1` is a read-modify-write, not an atomic step.

    Two threads reading the same value hand back the same id, and settle() then
    rewrites whichever event it finds first — correcting one call's window entry
    with another call's usage.
    """
    with _a_widened_race_window():
        for _ in range(TRIALS):
            limiter = TokenBudgetLimiter(tpm=1_000_000)
            granted, crashes = _reserve_from_threads(
                limiter, threads=12, tokens=100
            )
            assert not crashes, f"a reserving thread crashed: {crashes[0]!r}"
            assert len(granted) == 12, "every reservation fits this budget"
            assert len(set(granted)) == 12, f"duplicate reservation ids: {granted}"
