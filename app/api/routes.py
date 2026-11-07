import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.api.schemas import (
    ChatDoneEvent,
    ChatRequest,
    ChatResponse,
    question_token_count,
)
from app.agent.graph import ask, agent_graph, initial_state
from app.rag.vectorstore import replace_documents
from app.services.conversation import append_turn, load_conversation, save_conversation
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import (
    write_audit,
    source_breakdown,
    kb_gaps,
    daily_counts,
    recent_queries,
)

router = APIRouter(prefix="/api")
settings = get_settings()
logger = logging.getLogger(__name__)

@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: UUID):
    try:
        history = load_conversation(conversation_id)
    except RedisError as exc:
        logger.exception("Could not load conversation history")
        raise HTTPException(
            status_code=503,
            detail="Conversation memory is unavailable. Please try again.",
        ) from exc
    return {
        "conversation_id": str(conversation_id),
        "messages": [
            {
                "role": "user" if isinstance(message, HumanMessage) else "assistant",
                "content": message.content,
            }
            for message in history
            if isinstance(message, (HumanMessage, AIMessage))
        ],
    }


@router.get("/stats")
def stats(x_admin_key: str = Header(default="")):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    sources = source_breakdown()
    return {
        "total": sum(s["count"] for s in sources),
        "sources": sources,
        "daily": daily_counts(),
        "gaps": kb_gaps(),
        "recent": recent_queries(),
    }


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    conversation_id = payload.conversation_id or uuid4()
    started_at = perf_counter()
    logger.info(
        "Chat request started conversation_id=%s question_chars=%d question_tokens=%d",
        conversation_id,
        len(payload.question),
        question_token_count(payload.question),
    )
    try:
        history = load_conversation(conversation_id)
        result = ask(
            payload.question, history, request_id=str(conversation_id)
        )
        write_audit(payload.question, result["source_used"], result.get("trace", []))
        save_conversation(
            conversation_id,
            append_turn(history, payload.question, result["answer"]),
        )
        logger.info(
            "Chat request completed conversation_id=%s source=%s duration_ms=%d",
            conversation_id,
            result["source_used"],
            int((perf_counter() - started_at) * 1000),
        )

        return {
            "conversation_id": str(conversation_id),
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),

        }

    except RedisError as exc:
        logger.exception("Conversation memory is unavailable")
        raise HTTPException(
            status_code=503,
            detail="Conversation memory is unavailable. Please try again.",
        ) from exc
    except Exception as exc:
        logger.exception("Chat request failed conversation_id=%s", conversation_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest):
    conversation_id = payload.conversation_id or uuid4()

    def events():
        started_at = perf_counter()
        state = initial_state(
            payload.question,
            stream_tokens=True,
            request_id=str(conversation_id),
        )
        history: list[BaseMessage] = []
        emitted = 0
        audited = False
        logger.info(
            "Chat stream started conversation_id=%s question_chars=%d question_tokens=%d",
            conversation_id,
            len(payload.question),
            question_token_count(payload.question),
        )
        try:
            history = load_conversation(conversation_id)
            state = initial_state(
                payload.question,
                history,
                stream_tokens=True,
                request_id=str(conversation_id),
            )
            for mode, chunk in agent_graph.stream(
                state,
                stream_mode=["updates", "custom"],
            ):
                if mode == "custom":
                    if chunk.get("type") == "token":
                        yield _sse({"type": "token", "text": chunk["text"]})
                    continue
                for node, delta in chunk.items():
                    state.update(delta)
                    trace_lines = state.get("trace", [])
                    for line in trace_lines[emitted:]:
                        yield _sse({"type": "trace", "node": node, "text": line})
                    emitted = len(trace_lines)

            write_audit(payload.question, state["source_used"], state.get("trace", []))
            saved_messages = (
                append_turn(history, payload.question, state["answer"])
                if state["answer"]
                else [*history, HumanMessage(content=payload.question)]
            )
            save_conversation(
                conversation_id,
                saved_messages,
            )
            audited = True
            logger.info(
                "Chat stream completed conversation_id=%s source=%s tool_calls=%d duration_ms=%d",
                conversation_id,
                state["source_used"],
                state["tool_call_count"],
                int((perf_counter() - started_at) * 1000),
            )
            done_event = ChatDoneEvent(
                type="done",
                conversation_id=conversation_id,
                answer=state["answer"],
                source_used=state["source_used"],
                citations=state.get("citations", []),
                rewritten_query=state.get("current_query", payload.question),
            )
            yield _sse(done_event.model_dump(mode="json"))
        except RedisError:
            logger.exception("Conversation memory is unavailable")
            yield _sse({
                "type": "error",
                "conversation_id": str(conversation_id),
                "message": "Conversation memory is unavailable. Please try again.",
            })
        except Exception as exc:
            logger.exception("Streaming chat failed")
            yield _sse({
                "type": "error",
                "conversation_id": str(conversation_id),
                "message": str(exc),
            })
        finally:
            if not audited:
                logger.info(
                    "Chat stream ended before completion conversation_id=%s duration_ms=%d",
                    conversation_id,
                    int((perf_counter() - started_at) * 1000),
                )
            # client may disconnect mid-stream; keep the audit trail either way
            if not audited and state.get("answer"):
                write_audit(payload.question, state.get("source_used", ""), state.get("trace", []))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    dest.write_bytes(await file.read())
    docs = load_file(dest)
    chunks = chunk_documents(docs)
    if not chunks:
        raise HTTPException(status_code=422, detail="The uploaded document contains no indexable text")
    ids = replace_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}
