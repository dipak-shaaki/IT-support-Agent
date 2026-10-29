from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    current_query: str
    answer: str
    source_used: str
    tool_call_count: int
    private_kb_searched: bool
    web_searched: bool
    trace: list[str]
    citations: list[dict]
