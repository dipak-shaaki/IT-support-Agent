import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from app.rag import workflow


class FakeRetriever:
    def invoke(self, query):
        assert query == "connect to VPN"
        return [
            Document(
                page_content="Connect with the company VPN client.",
                metadata={"source": "vpn-guide.md"},
            )
        ]


class FakeBoundModel:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools

    def invoke(self, messages):
        self.model.available_tools.append({tool.name for tool in self.tools})
        return self.model.responses.pop(0)


class FakeModel:
    def __init__(self, responses):
        self.responses = responses
        self.available_tools = []

    def bind_tools(self, tools):
        return FakeBoundModel(self, tools)


def test_agent_calls_private_tool_before_it_can_call_web(monkeypatch):
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_private_kb",
                        "args": {"query": "connect to VPN"},
                        "id": "private-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_web",
                        "args": {"query": "current VPN client outage"},
                        "id": "web-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Use the company VPN client to connect."),
        ]
    )
    monkeypatch.setattr(workflow, "llm", lambda: model)
    monkeypatch.setattr(workflow, "get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(
        workflow,
        "web_search_provider",
        lambda: type(
            "FakeWebSearch",
            (),
            {
                "invoke": lambda self, request: {
                    "answer": "No current outage reported.",
                    "results": [
                        {
                            "title": "VPN service status",
                            "url": "https://status.example.test/vpn",
                            "content": "All systems operational.",
                        }
                    ],
                }
            },
        )(),
    )

    result = workflow.ask("How do I connect to VPN?")

    assert model.available_tools == [
        {"search_private_kb"},
        {"search_private_kb", "search_web"},
        {"search_private_kb", "search_web"},
    ]
    assert result["source_used"] == "web_search"
    assert result["answer"] == "Use the company VPN client to connect."
    assert result["current_query"] == "current VPN client outage"
    assert {citation["type"] for citation in result["citations"]} == {
        "private_kb",
        "web",
    }
    assert any("search_private_kb" in line for line in result["trace"])
    assert any("search_web" in line for line in result["trace"])


def test_private_search_tool_returns_documents_as_json(monkeypatch):
    monkeypatch.setattr(workflow, "get_retriever", lambda: FakeRetriever())

    result = json.loads(workflow.search_private_kb.invoke({"query": "connect to VPN"}))

    assert result == {
        "documents": [
            {
                "source": "vpn-guide.md",
                "content": "Connect with the company VPN client.",
            }
        ]
    }


def test_model_can_reply_without_calling_tools(monkeypatch):
    model = FakeModel([AIMessage(content="Hello! How can I help?")])
    monkeypatch.setattr(workflow, "llm", lambda: model)

    result = workflow.ask("Hello")

    assert result["answer"] == "Hello! How can I help?"
    assert result["source_used"] == "direct"
    assert result["citations"] == []
    assert result["trace"] == ["Direct response → no retrieval"]


def test_initial_state_includes_prior_conversation_turns():
    history = [
        HumanMessage(content="How do I connect to VPN?"),
        AIMessage(content="Use the company VPN client."),
    ]

    state = workflow.initial_state("What about on Mac?", history)

    assert state["messages"] == [
        *history,
        HumanMessage(content="What about on Mac?"),
    ]


def test_web_tool_call_before_private_search_is_rejected(monkeypatch):
    state = workflow.initial_state("How do I connect to VPN?")
    state["messages"].append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_web",
                    "args": {"query": "VPN"},
                    "id": "web-call",
                    "type": "tool_call",
                }
            ],
        )
    )

    with pytest.raises(ValueError, match="requested unavailable tools: search_web"):
        workflow.execute_tools(state)
