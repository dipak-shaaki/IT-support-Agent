import pytest
from pydantic import ValidationError

from app.api.schemas import (
    ChatDoneEvent,
    ChatRequest,
    ChatResponse,
    MAX_QUESTION_TOKENS,
    question_token_count,
)


def test_chat_request_accepts_exactly_400_tokens():
    question = ("hello " * MAX_QUESTION_TOKENS).strip()

    assert question_token_count(question) == MAX_QUESTION_TOKENS
    assert ChatRequest(question=f"  {question}  ").question == question


def test_chat_request_rejects_more_than_400_tokens():
    question = ("hello " * (MAX_QUESTION_TOKENS + 1)).strip()

    with pytest.raises(ValidationError, match="400-token limit"):
        ChatRequest(question=question)


@pytest.mark.parametrize("question", ["   ", "\n\t"])
def test_chat_request_rejects_blank_questions(question):
    with pytest.raises(ValidationError, match="must not be blank"):
        ChatRequest(question=question)


def test_chat_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        ChatRequest(question="")


def test_chat_request_preserves_minimum_question_length():
    with pytest.raises(ValidationError, match="at least two characters"):
        ChatRequest(question="?")


def test_chat_response_rejects_empty_or_unknown_output():
    base = {
        "conversation_id": "f0bb9e7a-7205-49c2-9daf-d4bcbe117c1a",
        "answer": "Here is the answer.",
        "source_used": "direct",
        "trace": [],
        "citations": [],
        "retrieval_query": "What is happening?",
    }

    assert ChatResponse.model_validate(base).answer == "Here is the answer."
    with pytest.raises(ValidationError):
        ChatResponse.model_validate({**base, "answer": ""})
    with pytest.raises(ValidationError):
        ChatResponse.model_validate({**base, "source_used": "unknown"})


def test_silent_done_event_allows_empty_answer_for_suppressed_turn():
    event = ChatDoneEvent(
        conversation_id="f0bb9e7a-7205-49c2-9daf-d4bcbe117c1a",
        answer="",
        source_used="direct",
        citations=[],
        retrieval_query="yes",
    )

    assert event.answer == ""
    assert event.type == "done"
