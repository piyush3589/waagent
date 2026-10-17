from waagent.ratelimit import RateLimiter


def test_allows_under_limit():
    rl = RateLimiter(per_minute=3)
    assert all(rl.allow("a") for _ in range(3))


def test_blocks_over_limit():
    rl = RateLimiter(per_minute=2)
    assert rl.allow("b") and rl.allow("b")
    assert rl.allow("b") is False


def test_keys_are_independent():
    rl = RateLimiter(per_minute=1)
    assert rl.allow("x")
    assert rl.allow("y")
    assert rl.allow("x") is False
