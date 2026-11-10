from uuid import UUID
from typing import Literal

import tiktoken
from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_TOKENS = 400
MAX_QUESTION_CHARACTERS = 12_000
_QUESTION_ENCODING = tiktoken.get_encoding("cl100k_base")


def question_token_count(question: str) -> int:
    return len(_QUESTION_ENCODING.encode(question, disallowed_special=()))


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARACTERS)
    conversation_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be blank")
        if len(question) < 2:
            raise ValueError("Question must contain at least two characters")

        token_count = question_token_count(question)
        if token_count > MAX_QUESTION_TOKENS:
            raise ValueError(
                f"Question exceeds the {MAX_QUESTION_TOKENS}-token limit "
                f"({token_count} tokens)"
            )
        return question


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str = Field(min_length=1)
    source_used: Literal["direct", "private_kb", "web_search"]
    trace: list[str]
    citations: list[dict[str, str]]
    retrieval_query: str


class ChatDoneEvent(BaseModel):
    type: Literal["done"] = "done"
    conversation_id: UUID
    answer: str
    source_used: Literal["direct", "private_kb", "web_search"]
    citations: list[dict[str, str]]
    retrieval_query: str
