import json
import logging
import re
import zipfile
from pathlib import Path
from uuid import uuid4
from time import perf_counter
from uuid import UUID

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
import anyio
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.core.metrics import CHAT_REQUESTS, INGEST_REQUESTS
from app.api.deps import (
    current_user,
    rate_limit,
    require_admin,
    require_ingestion_enabled,
)
from app.api.schemas import (
    ChatDoneEvent,
    ChatRequest,
    ChatResponse,
    question_token_count,
)
from app.agent.graph import ask, agent_graph, initial_state
from app.rag.vectorstore import get_bm25_index
from app.services.conversation import (
    append_turn,
    get_redis_client,
    load_conversation,
    save_conversation,
)
from app.services.ingest_jobs import create_job, get_job
from app.services.ingestion import EmptyDocumentError, index_file, SUPPORTED
from app.services.audit import (
    write_audit,
    source_breakdown,
    kb_gaps,
    gap_total,
    daily_counts,
    recent_queries,
)

router = APIRouter(prefix="/api")
settings = get_settings()
logger = logging.getLogger(__name__)


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.get("/ready")
def ready(response: Response):
    checks = {}
    try:
        get_redis_client().ping()
        checks["redis"] = "ok"
    except Exception:
        logger.warning("Readiness check failed for Redis", exc_info=True)
        checks["redis"] = "unavailable"
    try:
        checks["bm25"] = "ok" if len(get_bm25_index()) else "empty"
    except Exception:
        logger.warning("Readiness check failed for BM25", exc_info=True)
        checks["bm25"] = "unavailable"

    is_ready = checks["redis"] == "ok"
    if not is_ready:
        response.status_code = 503
    return {"status": "ready" if is_ready else "not_ready", "checks": checks}


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: UUID,
    user: str | None = Depends(current_user),
):
    try:
        history = load_conversation(conversation_id, user)
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
def stats(_: None = Depends(require_admin)):
    sources = source_breakdown()
    return {
        "total": sum(s["count"] for s in sources),
        "sources": sources,
        "daily": daily_counts(),
        "gaps": kb_gaps(),
        "gaps_total": gap_total(),
        "recent": recent_queries(),
    }


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    user: str | None = Depends(current_user),
    _: None = Depends(rate_limit),
):
    conversation_id = payload.conversation_id or uuid4()
    started_at = perf_counter()
    logger.info(
        "Chat request started conversation_id=%s question_chars=%d question_tokens=%d",
        conversation_id,
        len(payload.question),
        question_token_count(payload.question),
    )
    try:
        history = load_conversation(conversation_id, user)
        result = ask(
            payload.question, history, request_id=str(conversation_id)
        )
        write_audit(
            payload.question,
            result["source_used"],
            result.get("trace", []),
            result["answer"],
        )
        saved_messages = (
            append_turn(history, payload.question, result["answer"])
            if result["answer"]
            else [*history, HumanMessage(content=payload.question)]
        )
        save_conversation(conversation_id, saved_messages, user)
        logger.info(
            "Chat request completed conversation_id=%s source=%s duration_ms=%d",
            conversation_id,
            result["source_used"],
            int((perf_counter() - started_at) * 1000),
        )
        CHAT_REQUESTS.labels(result["source_used"], "ok").inc()

        return {
            "conversation_id": str(conversation_id),
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "retrieval_query": result.get("current_query", payload.question),
        }

    except RedisError as exc:
        CHAT_REQUESTS.labels("", "error").inc()
        logger.exception("Conversation memory is unavailable")
        raise HTTPException(
            status_code=503,
            detail="Conversation memory is unavailable. Please try again.",
        ) from exc
    except Exception as exc:
        CHAT_REQUESTS.labels("", "error").inc()
        logger.exception("Chat request failed conversation_id=%s", conversation_id)
        raise HTTPException(
            status_code=500,
            detail="The assistant could not complete this request. Please try again.",
        ) from exc


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat/stream")
def chat_stream(
    payload: ChatRequest,
    user: str | None = Depends(current_user),
    _: None = Depends(rate_limit),
):
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
            history = load_conversation(conversation_id, user)
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

            write_audit(
                payload.question,
                state["source_used"],
                state.get("trace", []),
                state["answer"],
            )
            saved_messages = (
                append_turn(history, payload.question, state["answer"])
                if state["answer"]
                else [*history, HumanMessage(content=payload.question)]
            )
            save_conversation(
                conversation_id,
                saved_messages,
                user,
            )
            audited = True
            logger.info(
                "Chat stream completed conversation_id=%s source=%s tool_calls=%d duration_ms=%d",
                conversation_id,
                state["source_used"],
                state["tool_call_count"],
                int((perf_counter() - started_at) * 1000),
            )
            CHAT_REQUESTS.labels(state["source_used"], "ok").inc()
            done_event = ChatDoneEvent(
                type="done",
                conversation_id=conversation_id,
                answer=state["answer"],
                source_used=state["source_used"],
                citations=state.get("citations", []),
                retrieval_query=state.get("current_query", payload.question),
            )
            yield _sse(done_event.model_dump(mode="json"))
        except RedisError:
            CHAT_REQUESTS.labels("", "error").inc()
            logger.exception("Conversation memory is unavailable")
            yield _sse({
                "type": "error",
                "conversation_id": str(conversation_id),
                "message": "Conversation memory is unavailable. Please try again.",
            })
        except Exception as exc:
            CHAT_REQUESTS.labels("", "error").inc()
            logger.exception("Streaming chat failed")
            yield _sse({
                "type": "error",
                "conversation_id": str(conversation_id),
                "message": "The assistant could not complete this request. Please try again.",
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
                write_audit(
                    payload.question,
                    state.get("source_used", ""),
                    state.get("trace", []),
                    state["answer"],
                )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



@router.post("/ingest")
async def ingest(
    _: None = Depends(require_ingestion_enabled),
    file: UploadFile = File(...),
    __: None = Depends(require_admin),
):
    original_filename = _safe_upload_filename(file.filename)
    suffix = Path(original_filename).suffix.lower()
    if suffix not in SUPPORTED:
        INGEST_REQUESTS.labels("rejected").inc()
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")
    upload_dir = Path(settings.upload_dir).resolve()
    await anyio.to_thread.run_sync(lambda: upload_dir.mkdir(parents=True, exist_ok=True))
    dest = (upload_dir / f"{uuid4().hex}{suffix}").resolve()
    if dest.parent != upload_dir:
        raise HTTPException(status_code=400, detail="Invalid upload destination")

    if settings.ingest_mode == "queue":
        saved = False
        try:
            await _save_upload(file, dest)
            saved = True
            await run_in_threadpool(_validate_upload_content, dest, suffix)
            job_id = await run_in_threadpool(create_job, original_filename, str(dest))
        except RedisError as exc:
            if saved:
                await _remove_upload(dest)
            INGEST_REQUESTS.labels("error").inc()
            logger.exception("Could not enqueue ingestion job file=%s", dest.name)
            raise HTTPException(
                status_code=503,
                detail="The ingestion queue is unavailable. Please try again.",
            ) from exc
        except Exception:
            if saved:
                await _remove_upload(dest)
            raise
        INGEST_REQUESTS.labels("queued").inc()
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": "queued", "file": original_filename},
        )

    saved = False
    try:
        await _save_upload(file, dest)
        saved = True
        await run_in_threadpool(_validate_upload_content, dest, suffix)
        chunks, ids = await run_in_threadpool(index_file, dest, original_filename)
    except EmptyDocumentError as exc:
        if saved:
            await _remove_upload(dest)
        INGEST_REQUESTS.labels("empty").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        if saved:
            await _remove_upload(dest)
        INGEST_REQUESTS.labels("rejected").inc()
        raise
    except Exception as exc:
        logger.exception("Document ingestion failed file=%s", original_filename)
        if saved:
            await _remove_upload(dest)
        INGEST_REQUESTS.labels("error").inc()
        raise HTTPException(
            status_code=500,
            detail="The document could not be indexed. Please try again.",
        ) from exc
    await _remove_upload(dest)
    INGEST_REQUESTS.labels("indexed").inc()
    return {"message": "Document indexed", "file": original_filename, "chunks": chunks, "ids_created": len(ids)}


@router.get("/ingest/jobs/{job_id}")
def ingest_job(
    job_id: str,
    _: None = Depends(require_ingestion_enabled),
    __: None = Depends(require_admin),
):
    try:
        job = get_job(job_id)
    except RedisError as exc:
        logger.exception("Could not read ingestion job %s", job_id)
        raise HTTPException(
            status_code=503,
            detail="Ingestion status is unavailable. Please try again.",
        ) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job


async def _save_upload(file: UploadFile, dest: Path) -> None:
    total = 0
    created = False
    try:
        async with await anyio.open_file(dest, "xb") as out:
            created = True
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "File exceeds the "
                            f"{settings.max_upload_bytes // (1024 * 1024)} MB upload limit"
                        ),
                    )
                await out.write(chunk)
    except BaseException:
        if created:
            await _remove_upload(dest)
        raise


async def _remove_upload(path: Path) -> None:
    await anyio.to_thread.run_sync(lambda: path.unlink(missing_ok=True))


def _safe_upload_filename(filename: str | None) -> str:
    """Keep a bounded display name as metadata; never use it as a path."""
    normalized = (filename or "document").replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    basename = re.sub(r"[\x00-\x1f\x7f]", "", basename).strip(" .")
    if not basename or basename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return basename[:255]


def _validate_upload_content(path: Path, suffix: str) -> None:
    """Check file signatures/content before handing uploads to document parsers."""
    with path.open("rb") as uploaded:
        prefix = uploaded.read(1024)
        if suffix == ".pdf":
            valid = b"%PDF-" in prefix
        elif suffix == ".docx":
            valid = prefix.startswith(b"PK\x03\x04")
            valid = valid and _is_docx(path)
        else:
            valid = b"\x00" not in prefix
            if valid:
                valid = _is_utf8_text(path)
    if not valid:
        raise HTTPException(status_code=400, detail="File content does not match its extension")


def _is_docx(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            return "[Content_Types].xml" in names and "word/document.xml" in names
    except (OSError, zipfile.BadZipFile):
        return False


def _is_utf8_text(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as uploaded:
            for line in uploaded:
                if "\x00" in line:
                    return False
        return True
    except (OSError, UnicodeDecodeError):
        return False
