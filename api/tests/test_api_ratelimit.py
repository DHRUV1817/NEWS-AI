from newsninja.api.ratelimit import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now


def test_requests_under_the_limit_are_allowed():
    clock = FakeClock()
    limiter = RateLimiter(limit=3, clock=clock.time)
    assert [limiter.check("1.2.3.4") for _ in range(3)] == [None, None, None]


def test_the_request_over_the_limit_is_refused_with_a_wait():
    clock = FakeClock()
    limiter = RateLimiter(limit=2, clock=clock.time)
    limiter.check("1.2.3.4")
    clock.now += 10.0
    limiter.check("1.2.3.4")
    assert limiter.check("1.2.3.4") == 50.0


def test_the_window_rolls_forward():
    clock = FakeClock()
    limiter = RateLimiter(limit=1, clock=clock.time)
    limiter.check("1.2.3.4")
    clock.now += 61.0
    assert limiter.check("1.2.3.4") is None


def test_clients_are_counted_separately():
    clock = FakeClock()
    limiter = RateLimiter(limit=1, clock=clock.time)
    limiter.check("1.2.3.4")
    assert limiter.check("5.6.7.8") is None


def test_idle_clients_are_evicted_rather_than_accumulating():
    """Without eviction the key map is an unbounded allocation per address.

    A caller cycling source addresses would grow it without limit, which turns
    the rate limiter into the memory-exhaustion vector it exists to prevent.
    """
    clock = FakeClock()
    limiter = RateLimiter(limit=1, clock=clock.time)
    for octet in range(200):
        limiter.check(f"10.0.0.{octet}")
    clock.now += 61.0
    limiter.check("10.0.1.1")
    assert limiter.tracked_clients() == 1


def test_eviction_is_throttled():
    """A stale entry must survive inside the throttle interval.

    This is the assertion that distinguishes a throttled sweep from an
    unthrottled one: at t=75 client "b" has been expired for 5 seconds, and
    only a sweep that was skipped leaves it in the map. Sweeping on every
    request would report 2 here.
    """
    clock = FakeClock()
    limiter = RateLimiter(limit=10, clock=clock.time)

    limiter.check("a")           # t=0 — the first sweep runs here
    clock.now = 10.0
    limiter.check("b")           # inside the interval, so no sweep
    clock.now = 60.0
    limiter.check("c")           # interval elapsed: sweep runs and drops "a"
    clock.now = 75.0
    limiter.check("d")           # inside the new interval: "b" is expired but must NOT be swept

    assert limiter.tracked_clients() == 3   # b (stale), c, d

    clock.now = 125.0
    limiter.check("e")           # crosses the boundary: the stale entry finally goes

    assert limiter.tracked_clients() == 2   # d, e
