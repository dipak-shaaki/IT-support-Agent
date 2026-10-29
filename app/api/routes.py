import json
import logging
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.rag.workflow import ask, agent_graph, initial_state
from app.rag.vectorstore import add_documents 
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

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)
    conversation_id: UUID | None = None


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


@router.post("/chat")
def chat(payload: ChatRequest):
    conversation_id = payload.conversation_id or uuid4()
    try:
        history = load_conversation(conversation_id)
        result = ask(payload.question, history)
        write_audit(payload.question, result["source_used"], result.get("trace", []))
        save_conversation(
            conversation_id,
            append_turn(history, payload.question, result["answer"]),
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
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest):
    conversation_id = payload.conversation_id or uuid4()

    def events():
        state = initial_state(payload.question, stream_tokens=True)
        history: list[BaseMessage] = []
        emitted = 0
        audited = False
        try:
            history = load_conversation(conversation_id)
            state = initial_state(payload.question, history, stream_tokens=True)
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
            save_conversation(
                conversation_id,
                append_turn(history, payload.question, state["answer"]),
            )
            audited = True
            yield _sse({
                "type": "done",
                "conversation_id": str(conversation_id),
                "answer": state["answer"],
                "source_used": state["source_used"],
                "citations": state.get("citations", []),
                "rewritten_query": state.get("current_query", payload.question),
            })
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
    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}
