import logging
import re
import time
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings
from app.services.conversation import get_redis_client

logger = logging.getLogger(__name__)

INGEST_QUEUE_KEY = "ingest:queue"
INGEST_LEASE_KEY = "ingest:leases"
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
            "attempts": "0",
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


def requeue_expired_jobs(now: float | None = None) -> int:
    """Return jobs whose worker lease expired; give up after max attempts."""
    client = get_redis_client()
    now = time.time() if now is None else now
    settings = get_settings()
    requeued = 0
    for job_id in client.zrangebyscore(INGEST_LEASE_KEY, "-inf", now):
        key = _job_key(job_id)
        raw = client.hgetall(key)
        if not raw or raw.get("status") != "running":
            client.zrem(INGEST_LEASE_KEY, job_id)
            continue

        attempts = int(raw.get("attempts", 0) or 0) + 1
        if attempts >= settings.ingest_max_attempts:
            _discard_upload(raw.get("path", ""))
            mark_failed(
                job_id,
                "Ingestion job exceeded its retry limit after the worker lease expired",
            )
            logger.error(
                "Ingestion job abandoned after %d attempts job_id=%s file=%s",
                attempts,
                job_id,
                raw.get("file", ""),
            )
            continue

        client.hset(key, mapping={"status": "queued", "attempts": str(attempts)})
        client.rpush(INGEST_QUEUE_KEY, job_id)
        client.zrem(INGEST_LEASE_KEY, job_id)
        logger.warning(
            "Requeued stalled ingestion job job_id=%s file=%s attempt=%d",
            job_id,
            raw.get("file", ""),
            attempts,
        )
        requeued += 1
    return requeued


def claim_job(timeout: int = 5) -> dict | None:
    """Block for the next queued job and mark it running with a lease."""
    requeue_expired_jobs()
    client = get_redis_client()
    item = client.blpop(INGEST_QUEUE_KEY, timeout=timeout)
    if not item:
        return None
    job_id = item[1]
    raw = client.hgetall(_job_key(job_id))
    if not raw or raw.get("status") != "queued":
        return None

    now = time.time()
    lease_until = now + get_settings().ingest_lease_seconds
    client.hset(
        _job_key(job_id),
        mapping={
            "status": "running",
            "claimed_at": str(int(now)),
            "lease_until": str(int(lease_until)),
        },
    )
    client.zadd(INGEST_LEASE_KEY, {job_id: lease_until})
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
    client.zrem(INGEST_LEASE_KEY, job_id)
    client.expire(_job_key(job_id), get_settings().ingest_job_ttl_seconds)


def mark_failed(job_id: str, error: str) -> None:
    client = get_redis_client()
    client.hset(
        _job_key(job_id),
        mapping={"status": "error", "error": error[:500]},
    )
    client.zrem(INGEST_LEASE_KEY, job_id)
    client.expire(_job_key(job_id), get_settings().ingest_job_ttl_seconds)


def _discard_upload(path: str) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove upload for failed job path=%s", path, exc_info=True)
