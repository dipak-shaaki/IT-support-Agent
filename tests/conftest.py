import pytest

from app.api import deps


class FakeRedis:
    def __init__(self):
        self.counts = {}

    def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, key, seconds):
        pass


@pytest.fixture(autouse=True)
def isolated_rate_limit_redis(monkeypatch):
    """Keep unit tests from opening a real Redis connection through the limiter."""
    monkeypatch.setattr(deps, "get_redis_client", lambda: FakeRedis())
