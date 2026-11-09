import re
import time
from uuid import uuid4

from app.core.config import get_settings
from app.services.conversation import get_redis_client

INGEST_QUEUE_KEY = "ingest:queue"
JOB_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


def _job_key(job_id: str) -> str:
    return f"ingest:job:{job_id}"


def _public(job_id: str, raw: dict) -> dict:
    return {
        "job_id": job_id,
        "status": raw.get("status", "queued"),
        "file": raw.get("file", ""),
        "chunks": int(raw.get("chunks", 0) or 0),
        "ids_created": int(raw.get("ids_created", 0) or 0),
        "error": raw.get("error", ""),
    }


def create_job(filename: str, path: str) -> str:
    """Record the job before enqueueing it so a status lookup never misses it."""
    job_id = uuid4().hex
    client = get_redis_client()
    key = _job_key(job_id)
    client.hset(
        key,
        mapping={
            "status": "queued",
            "file": filename,
            "path": path,
            "created_at": str(int(time.time())),
        },
    )
    client.expire(key, get_settings().ingest_job_ttl_seconds)
    client.rpush(INGEST_QUEUE_KEY, job_id)
    return job_id


def get_job(job_id: str) -> dict | None:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        return None
    raw = get_redis_client().hgetall(_job_key(job_id))
    return _public(job_id, raw) if raw else None


def claim_job(timeout: int = 5) -> dict | None:
    """Block for the next queued job and mark it running."""
    # shortcut: a worker crash after claiming leaves the job "running" and its upload on disk;
    # add a lease/requeue if ingestion must survive worker failures.
    item = get_redis_client().blpop(INGEST_QUEUE_KEY, timeout=timeout)
    if not item:
        return None
    job_id = item[1]
    raw = get_redis_client().hgetall(_job_key(job_id))
    if not raw or raw.get("status") != "queued":
        return None
    get_redis_client().hset(_job_key(job_id), mapping={"status": "running"})
    return {"job_id": job_id, "file": raw.get("file", ""), "path": raw.get("path", "")}


def mark_done(job_id: str, chunks: int, ids_created: int) -> None:
    client = get_redis_client()
    client.hset(
        _job_key(job_id),
        mapping={
            "status": "done",
            "chunks": str(chunks),
            "ids_created": str(ids_created),
        },
    )
    client.expire(_job_key(job_id), get_settings().ingest_job_ttl_seconds)


def mark_failed(job_id: str, error: str) -> None:
    client = get_redis_client()
    client.hset(
        _job_key(job_id),
        mapping={"status": "error", "error": error[:500]},
    )
    client.expire(_job_key(job_id), get_settings().ingest_job_ttl_seconds)
