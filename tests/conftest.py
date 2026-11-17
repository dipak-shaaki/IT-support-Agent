import pytest

from app.api import deps
from app.agent import tools as agent_tools


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


class EmptyRetriever:
    def invoke(self, _query):
        return []


@pytest.fixture(autouse=True)
def isolated_private_kb(monkeypatch):
    """Keep unit tests independent of Pinecone and the configured corpus."""
    from app import main as app_main

    monkeypatch.setattr(agent_tools, "get_retriever", lambda: EmptyRetriever())
    monkeypatch.setattr(app_main, "get_embeddings", lambda: None)
    monkeypatch.setattr(app_main, "get_retriever", lambda: EmptyRetriever())
