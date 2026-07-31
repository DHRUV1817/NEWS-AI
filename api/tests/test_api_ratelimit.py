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
    """Eviction sweep only runs once per window, not on every request.

    This keeps the operation O(1) amortised per request rather than O(n) in
    the number of tracked clients. Many distinct clients seen within one window
    remain tracked until after the window elapses and a new request triggers
    the sweep.
    """
    clock = FakeClock()
    limiter = RateLimiter(limit=1, clock=clock.time)
    # See many distinct clients at time 0
    for octet in range(20):
        limiter.check(f"10.0.0.{octet}")
    assert limiter.tracked_clients() == 20
    # At time 30, they're still tracked (sweep hasn't run; < 60 seconds since last sweep at time 0)
    clock.now += 30.0
    limiter.check("10.0.1.0")
    assert limiter.tracked_clients() == 21
    # At time 90, sweep finally runs and clears the old clients
    # (60+ seconds since the first sweep at time 0)
    clock.now += 60.0
    limiter.check("10.0.2.0")
    # Old clients from times 0 and 30 are now gone; only the latest remains
    assert limiter.tracked_clients() == 1
