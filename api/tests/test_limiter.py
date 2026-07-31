import pytest

from newsninja.analysis.limiter import TokenBudgetLimiter


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
