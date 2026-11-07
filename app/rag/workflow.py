"""Compatibility exports for the agent modules now located in :mod:`app.agent`."""

from app.agent.graph import agent_graph, ask, build_graph, initial_state
from app.agent.model import (
    content_text as _content_text,
    fallback_llms,
    invoke_with_fallbacks as _invoke_with_fallbacks,
    llm,
    stream_with_fallbacks as _stream_with_fallbacks,
)
from app.agent.nodes import (
    MAX_TOOL_CALLS,
    call_agent,
    execute_tools,
    finalize,
    mandatory_private_search,
    mandatory_web_search,
    route_after_agent,
)
from app.agent.policy import CURRENT_STATUS_PATTERN
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import (
    TOOLS,
    search_private_kb,
    search_web,
    web_search_provider,
)
