import json
from functools import lru_cache
from uuid import UUID

import redis
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)

from app.core.config import get_settings

MAX_HISTORY_MESSAGES = 20


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=5,
    )


def _conversation_key(conversation_id: UUID) -> str:
    return f"conversation:{conversation_id}"


def load_conversation(conversation_id: UUID) -> list[BaseMessage]:
    serialized = get_redis_client().get(_conversation_key(conversation_id))
    if serialized is None:
        return []
    messages = messages_from_dict(json.loads(serialized))
    return [
        message
        for message in messages
        if isinstance(message, HumanMessage)
        or (isinstance(message, AIMessage) and not message.tool_calls)
    ]


def save_conversation(conversation_id: UUID, messages: list[BaseMessage]) -> None:
    settings = get_settings()
    retained = messages[-MAX_HISTORY_MESSAGES:]
    get_redis_client().set(
        _conversation_key(conversation_id),
        json.dumps(messages_to_dict(retained)),
        ex=settings.conversation_ttl_seconds,
    )


def append_turn(
    history: list[BaseMessage],
    question: str,
    answer: str,
) -> list[BaseMessage]:
    return [
        *history,
        HumanMessage(content=question),
        AIMessage(content=answer),
    ][-MAX_HISTORY_MESSAGES:]
