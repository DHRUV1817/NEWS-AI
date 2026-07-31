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
