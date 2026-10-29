from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage

from app.services import conversation


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.expiry = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex):
        self.data[key] = value
        self.expiry[key] = ex


def test_conversation_round_trip_uses_configured_expiry(monkeypatch):
    conversation_id = UUID("f0bb9e7a-7205-49c2-9daf-d4bcbe117c1a")
    client = FakeRedis()
    monkeypatch.setattr(conversation, "get_redis_client", lambda: client)
    monkeypatch.setattr(
        conversation,
        "get_settings",
        lambda: type("Settings", (), {"conversation_ttl_seconds": 3600})(),
    )
    messages = [
        HumanMessage(content="How do I connect to VPN?"),
        AIMessage(content="Use the company VPN client."),
    ]

    conversation.save_conversation(conversation_id, messages)
    loaded = conversation.load_conversation(conversation_id)

    assert loaded == messages
    assert client.expiry[f"conversation:{conversation_id}"] == 3600


def test_missing_conversation_returns_empty_history(monkeypatch):
    monkeypatch.setattr(conversation, "get_redis_client", lambda: FakeRedis())

    assert conversation.load_conversation(UUID(int=1)) == []


def test_append_turn_keeps_only_recent_message_window():
    history = [
        HumanMessage(content=f"question {index}")
        for index in range(conversation.MAX_HISTORY_MESSAGES)
    ]

    updated = conversation.append_turn(history, "follow up", "answer")

    assert len(updated) == conversation.MAX_HISTORY_MESSAGES
    assert updated[-2:] == [
        HumanMessage(content="follow up"),
        AIMessage(content="answer"),
    ]
