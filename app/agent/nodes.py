import json
import logging
from typing import Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolNode

from app.agent import model, prompts
from app.agent.policy import (
    is_answer_source_followup,
    is_non_retrieval_turn,
)
from app.agent.state import AgentState
from app.agent.tools import search_private_kb, search_web

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 6
REPETITIVE_FOLLOWUP_ANSWER = (
    "I don't have another question to ask. If you describe the issue or the error "
    "you're seeing, I'll help you troubleshoot it."
)


def available_tools(state: AgentState) -> list:
    if (
        is_non_retrieval_turn(state["question"])
        and not state["require_web_search"]
    ):
        return []
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


def route_initial_state(
    state: AgentState,
) -> Literal["agent", "followup_guard", "prefetch_private_kb"]:
    if state["repetitive_followup"]:
        return "followup_guard"
    if (
        not state["require_web_search"]
        and not is_non_retrieval_turn(state["question"])
    ):
        return "prefetch_private_kb"
    return "agent"


def _retrieval_query(state: AgentState) -> str:
    question = state["question"]
    if not is_answer_source_followup(question):
        return question

    for message in reversed(state["messages"][:-1]):
        if (
            not isinstance(message, HumanMessage)
            or not isinstance(message.content, str)
        ):
            continue
        candidate = message.content.strip()
        if (
            candidate
            and not is_non_retrieval_turn(candidate)
            and not is_answer_source_followup(candidate)
        ):
            return candidate
    return question


def stop_repetitive_followup(_state: AgentState):
    return {
        "answer": REPETITIVE_FOLLOWUP_ANSWER,
        "source_used": "direct",
    }


def prefetch_private_kb(state: AgentState):
    """Prefetch private KB to avoid an extra LLM tool-decision round-trip."""
    query = _retrieval_query(state)
    call_id = str(uuid4())
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": search_private_kb.name,
                "args": {"query": query},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )
    result = search_private_kb.invoke({"query": query})
    tool_message = ToolMessage(
        content=result,
        name=search_private_kb.name,
        tool_call_id=call_id,
    )

    trace = [
        *state["trace"],
        f"Tool call → {search_private_kb.name}({query})",
    ]
    citations = list(state["citations"])
    payload = _decode_tool_result(tool_message, search_private_kb.name)
    if payload is not None:
        _add_private_citations(payload, citations, trace)

    return {
        "messages": [call, tool_message],
        "trace": trace,
        "citations": citations,
        "private_kb_searched": True,
        "current_query": query,
        "tool_call_count": state["tool_call_count"] + 1,
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
        llm_metrics = list(state.get("llm_metrics", []))
        response = model.stream_with_fallbacks(
            models,
            messages,
            writer,
            state["request_id"],
            metrics_callback=llm_metrics.append,
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
                if (
                    call["name"] == search_private_kb.name
                    and state["private_kb_searched"]
                    and query.strip().casefold()
                    == state["current_query"].strip().casefold()
                ):
                    continue
                trace = [*trace, f"Tool call → {call['name']}({query})"]
    update = {"messages": [response], "trace": trace}
    if state["stream_tokens"]:
        update["llm_metrics"] = llm_metrics
    return update


def route_after_agent(
    state: AgentState,
) -> Literal["tools", "mandatory_private_search", "mandatory_web_search", "finalize"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return _mandatory_step(state) or "finalize"
    available_names = {tool.name for tool in available_tools(state)}
    if any(call["name"] not in available_names for call in last_message.tool_calls):
        return "finalize"
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

    previous_kb_result = next(
        (
            message
            for message in reversed(state["messages"])
            if isinstance(message, ToolMessage)
            and message.name == search_private_kb.name
        ),
        None,
    )
    cached_calls = {
        call["id"]
        for call in response.tool_calls
        if (
            state["private_kb_searched"]
            and previous_kb_result is not None
            and call["name"] == search_private_kb.name
            and call.get("args", {}).get("query", "").strip().casefold()
            == state["current_query"].strip().casefold()
        )
    }
    calls_to_run = [
        call for call in response.tool_calls if call["id"] not in cached_calls
    ]
    results_by_id = {}
    if calls_to_run:
        execution_response = response.model_copy(update={"tool_calls": calls_to_run})
        result = ToolNode(tools).invoke({"messages": [execution_response]})
        results_by_id.update(
            {message.tool_call_id: message for message in result["messages"]}
        )
    for call in response.tool_calls:
        if call["id"] in cached_calls:
            results_by_id[call["id"]] = ToolMessage(
                content=previous_kb_result.content,
                name=search_private_kb.name,
                tool_call_id=call["id"],
            )
    result_messages = [results_by_id[call["id"]] for call in response.tool_calls]
    updates = _process_tool_results(calls_to_run, results_by_id, state)
    return {
        "messages": result_messages,
        "tool_call_count": state["tool_call_count"] + len(response.tool_calls),
        **updates,
    }


def _process_tool_results(calls, results_by_id, state):
    citations = list(state["citations"])
    private_kb_searched = state["private_kb_searched"]
    web_searched = state["web_searched"]
    current_query = state["current_query"]
    trace = list(state["trace"])
    for call in calls:
        current_query = call.get("args", {}).get("query", current_query)
        tool_message = results_by_id.get(call["id"])
        if not isinstance(tool_message, ToolMessage):
            continue
        payload = _decode_tool_result(tool_message, call["name"])
        if payload is None:
            continue
        if call["name"] == search_private_kb.name:
            private_kb_searched = True
            citations = [
                citation
                for citation in citations
                if citation.get("type") != "private_kb"
            ]
            _add_private_citations(payload, citations, trace)
        elif call["name"] == search_web.name:
            web_searched = True
            _add_web_citations(payload, citations, trace)
    return {
        "private_kb_searched": private_kb_searched,
        "web_searched": web_searched,
        "current_query": current_query,
        "citations": citations,
        "trace": trace,
    }


def _decode_tool_result(message, name):
    try:
        return json.loads(model.content_text(message.content))
    except (TypeError, json.JSONDecodeError):
        logger.exception("Could not decode result from tool %s", name)
        return None


def _add_private_citations(payload, citations, trace):
    documents = payload.get("documents", [])
    trace.append(f"Private KB results → {len(documents)} chunks")
    for document in documents:
        source = document.get("source", "Private KB")
        citation = {"title": document.get("record_id") or source.rsplit("/", 1)[-1], "url": "", "type": "private_kb"}
        if citation not in citations:
            citations.append(citation)


def _add_web_citations(payload, citations, trace):
    results = payload.get("results", [])
    trace.append(f"Web results → {len(results)} pages")
    for item in results:
        url = item.get("url", "")
        citation = {"title": item.get("title") or url, "url": url, "type": "web"}
        if url and citation not in citations:
            citations.append(citation)


def finalize(state: AgentState):
    messages = list(state["messages"])
    last_message = messages[-1]
    trace = list(state["trace"])
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        answer = model.content_text(last_message.content).strip()
    else:
        answer = _generate_final_answer(messages, last_message, state)
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


def _generate_final_answer(messages, last_message, state):
    completed = messages[:-1] if isinstance(last_message, AIMessage) else messages
    limit_reached = state["tool_call_count"] >= MAX_TOOL_CALLS
    instruction = (
        f"The tool-call limit ({MAX_TOOL_CALLS}) has been reached. "
        if limit_reached else "No additional tools are available. "
    )
    prompt = [
        SystemMessage(
            content=(
                f"{prompts.SYSTEM_PROMPT}\n{instruction}"
                "Answer using only the completed tool results."
            )
        ),
        *completed,
    ]
    providers = [model.llm(), *model.fallback_llms()]
    if state["stream_tokens"]:
        response = model.stream_with_fallbacks(
            providers, prompt, get_stream_writer(), state["request_id"]
        )
    else:
        response = model.invoke_with_fallbacks(providers, prompt, state["request_id"])
    answer = model.content_text(response.content).strip()
    return answer
