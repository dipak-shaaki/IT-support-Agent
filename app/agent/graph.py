from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agent import nodes
from app.agent.policy import CURRENT_STATUS_PATTERN, has_repetitive_follow_up
from app.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", nodes.call_agent)
    graph.add_node("mandatory_private_search", nodes.mandatory_private_search)
    graph.add_node("mandatory_web_search", nodes.mandatory_web_search)
    graph.add_node("tools", nodes.execute_tools)
    graph.add_node("finalize", nodes.finalize)
    graph.add_node("followup_guard", nodes.stop_repetitive_followup)
    graph.add_conditional_edges(
        START,
        nodes.route_initial_state,
        {
            "agent": "agent",
            "followup_guard": "followup_guard",
        },
    )
    graph.add_conditional_edges(
        "agent",
        nodes.route_after_agent,
        {
            "tools": "tools",
            "mandatory_private_search": "mandatory_private_search",
            "mandatory_web_search": "mandatory_web_search",
            "finalize": "finalize",
        },
    )
    graph.add_edge("mandatory_private_search", "tools")
    graph.add_edge("mandatory_web_search", "tools")
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    graph.add_edge("followup_guard", END)
    return graph.compile()


agent_graph = build_graph()


def initial_state(
    question: str,
    history: list[AnyMessage] | None = None,
    stream_tokens: bool = False,
    request_id: str = "",
) -> AgentState:
    return {
        "messages": [*(history or []), HumanMessage(content=question)],
        "question": question,
        "request_id": request_id,
        "repetitive_followup": has_repetitive_follow_up(
            question, history or []
        ),
        "current_query": question,
        "answer": "",
        "source_used": "",
        "tool_call_count": 0,
        "stream_tokens": stream_tokens,
        "require_web_search": bool(CURRENT_STATUS_PATTERN.search(question)),
        "private_kb_searched": False,
        "web_searched": False,
        "trace": [],
        "citations": [],
    }


def ask(
    question: str,
    history: list[AnyMessage] | None = None,
    request_id: str = "",
):
    return agent_graph.invoke(initial_state(question, history, request_id=request_id))
