import pytest

from app.api import deps


class FakeRedis:
    def __init__(self):
        self.zsets = {}

    def zremrangebyscore(self, key, low, high):
        self.zsets[key] = {
            member: score
            for member, score in self.zsets.get(key, {}).items()
            if score > high
        }

    def zcard(self, key):
        return len(self.zsets.get(key, {}))

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    def expire(self, key, seconds):
        pass


@pytest.fixture(autouse=True)
def isolated_rate_limit_redis(monkeypatch):
    """Keep unit tests from opening a real Redis connection through the limiter."""
    monkeypatch.setattr(deps, "get_redis_client", lambda: FakeRedis())
