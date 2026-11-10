import json
import logging
from typing import Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolNode

from app.agent import model, prompts
from app.agent.state import AgentState
from app.agent.tools import search_private_kb, search_web

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 6
REPETITIVE_FOLLOWUP_ANSWER = (
    "I don't have another question to ask. If you describe the issue or the error "
    "you're seeing, I'll help you troubleshoot it."
)


def available_tools(state: AgentState) -> list:
    tools = [search_private_kb]
    if state["require_web_search"] and state["private_kb_searched"]:
        tools.append(search_web)
    return tools


def _mandatory_step(state: AgentState) -> str | None:
    if state["require_web_search"] and not state["private_kb_searched"]:
        return "mandatory_private_search"
    if state["require_web_search"] and not state["web_searched"]:
        return "mandatory_web_search"
    return None


def route_initial_state(state: AgentState) -> Literal["agent", "followup_guard"]:
    return "followup_guard" if state["repetitive_followup"] else "agent"


def stop_repetitive_followup(state: AgentState):
    return {
        "answer": REPETITIVE_FOLLOWUP_ANSWER,
        "source_used": "direct",
    }


def call_agent(state: AgentState):
    tools = available_tools(state)
    models = [
        model.llm().bind_tools(tools),
        *(fallback.bind_tools(tools) for fallback in model.fallback_llms()),
    ]
    messages = [SystemMessage(content=prompts.SYSTEM_PROMPT), *state["messages"]]
    if state["stream_tokens"]:
        writer = get_stream_writer()
        response = model.stream_with_fallbacks(
            models, messages, writer, state["request_id"]
        )
    else:
        response = model.invoke_with_fallbacks(
            models, messages, state["request_id"]
        )
    trace = state["trace"]
    if response.tool_calls:
        for call in response.tool_calls:
            if call["name"] in {tool.name for tool in tools}:
                query = call.get("args", {}).get("query", "")
                trace = [*trace, f"Tool call → {call['name']}({query})"]
    return {"messages": [response], "trace": trace}


def route_after_agent(
    state: AgentState,
) -> Literal["tools", "mandatory_private_search", "mandatory_web_search", "finalize"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return _mandatory_step(state) or "finalize"
    if any(
        call["name"] == search_web.name
        for call in last_message.tool_calls
    ) and not (state["require_web_search"] and state["private_kb_searched"]):
        return "finalize"
    if state["tool_call_count"] >= MAX_TOOL_CALLS:
        return _mandatory_step(state) or "finalize"
    return "tools"


def mandatory_private_search(state: AgentState):
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": search_private_kb.name,
                "args": {"query": state["question"]},
                "id": str(uuid4()),
                "type": "tool_call",
            }
        ],
    )
    return {
        "messages": [call],
        "trace": [
            *state["trace"],
            "Current service-status question → private KB lookup required first",
            f"Tool call → {search_private_kb.name}({state['question']})",
        ],
    }


def mandatory_web_search(state: AgentState):
    query = state["question"]
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": search_web.name,
                "args": {"query": query},
                "id": str(uuid4()),
                "type": "tool_call",
            }
        ],
    )
    return {
        "messages": [call],
        "trace": [
            *state["trace"],
            "Current service-status question → web lookup required",
            f"Tool call → {search_web.name}({query})",
        ],
    }


def execute_tools(state: AgentState):
    response = state["messages"][-1]
    if not isinstance(response, AIMessage):
        raise TypeError("Tool execution requires an assistant tool-call message")

    tools = available_tools(state)
    available_names = {item.name for item in tools}
    unavailable = {
        call["name"] for call in response.tool_calls
    } - available_names
    if unavailable:
        raise ValueError(f"Model requested unavailable tools: {', '.join(sorted(unavailable))}")

    result = ToolNode(tools).invoke({"messages": [response]})
    citations = list(state["citations"])
    private_kb_searched = state["private_kb_searched"]
    web_searched = state["web_searched"]
    current_query = state["current_query"]
    trace = list(state["trace"])

    results_by_id = {message.tool_call_id: message for message in result["messages"]}
    for call in response.tool_calls:
        current_query = call.get("args", {}).get("query", current_query)
        tool_message = results_by_id.get(call["id"])
        if not isinstance(tool_message, ToolMessage):
            continue
        try:
            payload = json.loads(model.content_text(tool_message.content))
        except (TypeError, json.JSONDecodeError):
            logger.exception("Could not decode result from tool %s", call["name"])
            continue

        if call["name"] == search_private_kb.name:
            private_kb_searched = True
            documents = payload.get("documents", [])
            trace.append(f"Private KB results → {len(documents)} chunks")
            for document in documents:
                source = document.get("source", "Private KB")
                citation = {
                    "title": document.get("record_id")
                    or source.rsplit("/", 1)[-1],
                    "url": "",
                    "type": "private_kb",
                }
                if citation not in citations:
                    citations.append(citation)
        elif call["name"] == search_web.name:
            web_searched = True
            results = payload.get("results", [])
            trace.append(f"Web results → {len(results)} pages")
            for item in results:
                url = item.get("url", "")
                citation = {
                    "title": item.get("title") or url,
                    "url": url,
                    "type": "web",
                }
                if url and citation not in citations:
                    citations.append(citation)

    return {
        "messages": result["messages"],
        "tool_call_count": state["tool_call_count"] + len(response.tool_calls),
        "private_kb_searched": private_kb_searched,
        "web_searched": web_searched,
        "current_query": current_query,
        "citations": citations,
        "trace": trace,
    }


def finalize(state: AgentState):
    messages = list(state["messages"])
    last_message = messages[-1]
    trace = list(state["trace"])
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        answer = model.content_text(last_message.content).strip()
    else:
        completed_messages = messages[:-1] if isinstance(last_message, AIMessage) else messages
        prompt = [
            SystemMessage(
                content=(
                    f"{prompts.SYSTEM_PROMPT}\n"
                    + (
                        f"The tool-call limit ({MAX_TOOL_CALLS}) has been reached. "
                        if state["tool_call_count"] >= MAX_TOOL_CALLS
                        else "No additional tools are available. "
                    )
                    + "Answer using only the completed tool results."
                )
            ),
            *completed_messages,
        ]
        if state["stream_tokens"]:
            answer = ""
            writer = get_stream_writer()
            response = model.stream_with_fallbacks(
                [model.llm(), *model.fallback_llms()],
                prompt,
                writer,
                state["request_id"],
            )
            answer = model.content_text(response.content).strip()
        else:
            answer_message = model.invoke_with_fallbacks(
                [model.llm(), *model.fallback_llms()],
                prompt,
                state["request_id"],
            )
            answer = model.content_text(answer_message.content).strip()
        trace.append(
            "Tool-call limit reached → finalized from completed results"
            if state["tool_call_count"] >= MAX_TOOL_CALLS
            else "Unavailable tool request ignored → finalized from completed results"
        )

    if not answer:
        raise RuntimeError("The assistant returned an empty answer")

    if state["web_searched"]:
        source_used = "web_search"
        trace_label = "Answer generation → WEB SEARCH"
    elif state["private_kb_searched"]:
        source_used = "private_kb"
        trace_label = "Answer generation → PRIVATE KB"
    else:
        source_used = "direct"
        trace_label = "Direct response → no retrieval"

    return {
        "answer": answer,
        "source_used": source_used,
        "trace": [*trace, trace_label],
    }
