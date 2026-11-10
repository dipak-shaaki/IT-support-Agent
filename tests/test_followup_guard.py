from langchain_core.messages import AIMessage, HumanMessage

from app.agent import graph as agent_graph
from app.agent import nodes


def test_third_consecutive_acknowledgment_uses_normal_graph(monkeypatch):
    history = [
        HumanMessage(content="yes"),
        AIMessage(content="Can you share the error?"),
        HumanMessage(content="okay"),
        AIMessage(content="What does the message say?"),
    ]

    def unexpected_model_call(_state):
        return {"messages": [AIMessage(content="A normal reply.")]}

    monkeypatch.setattr(nodes, "call_agent", unexpected_model_call)
    graph = agent_graph.build_graph()

    result = graph.invoke(agent_graph.initial_state("yeah", history))

    assert result["repetitive_followup"] is False
    assert result["source_used"] == "direct"
    assert result["answer"] == "A normal reply."


def test_fourth_consecutive_acknowledgment_returns_a_prompt_and_skips_model(monkeypatch):
    history = [
        HumanMessage(content="yes"),
        AIMessage(content="Can you share the error?"),
        HumanMessage(content="okay"),
        AIMessage(content="What does the message say?"),
        HumanMessage(content="yes"),
        AIMessage(content="Anything else you can share?"),
    ]

    def unexpected_model_call(_state):
        raise AssertionError("The graph should not call the model for repeated acknowledgments")

    monkeypatch.setattr(nodes, "call_agent", unexpected_model_call)
    graph = agent_graph.build_graph()

    result = graph.invoke(agent_graph.initial_state("yeah", history))

    assert result["repetitive_followup"] is True
    assert result["answer"] == nodes.REPETITIVE_FOLLOWUP_ANSWER
    assert result["source_used"] == "direct"
    assert result["trace"] == []


def test_specific_question_resumes_normal_graph_after_acknowledgments(monkeypatch):
    history = [
        HumanMessage(content="yes"),
        AIMessage(content="Can you share the error?"),
        HumanMessage(content="okay"),
        AIMessage(content="What does the message say?"),
    ]
    model_questions = []

    def answer_question(state):
        model_questions.append(state["question"])
        return {"messages": [AIMessage(content="Tell me the exact error message.")]}

    monkeypatch.setattr(nodes, "call_agent", answer_question)
    graph = agent_graph.build_graph()

    result = graph.invoke(
        agent_graph.initial_state("VPN keeps disconnecting", history)
    )

    assert result["repetitive_followup"] is False
    assert model_questions == ["VPN keeps disconnecting"]
    assert result["answer"] == "Tell me the exact error message."


def test_two_acknowledgments_do_not_activate_guard():
    history = [
        HumanMessage(content="yes"),
        AIMessage(content="Can you share the error?"),
    ]

    state = agent_graph.initial_state("okay", history)

    assert state["repetitive_followup"] is False


def test_specific_question_resets_repetitive_acknowledgment_count():
    history = [
        HumanMessage(content="yes"),
        AIMessage(content="Can you share the error?"),
        HumanMessage(content="okay"),
        AIMessage(content="What does the message say?"),
        HumanMessage(content="yes"),
        AIMessage(content="Anything else you can share?"),
    ]

    state = agent_graph.initial_state("The VPN disconnects every few minutes", history)

    assert state["repetitive_followup"] is False
