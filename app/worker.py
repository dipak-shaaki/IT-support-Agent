import logging
import signal
import time
from pathlib import Path

from redis.exceptions import RedisError

from app.core.logging import configure_logging
from app.services.ingest_jobs import claim_job, mark_done, mark_failed
from app.services.ingestion import index_file

logger = logging.getLogger(__name__)

_stopping = False
MAX_BACKOFF_SECONDS = 30


def _stop(_signum, _frame) -> None:
    global _stopping
    _stopping = True


def _wait(seconds: float) -> None:
    """Sleep in short steps so SIGTERM still stops the worker promptly."""
    deadline = time.monotonic() + seconds
    while not _stopping and time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def process_job(job: dict) -> None:
    if not job.get("path"):
        mark_failed(job["job_id"], "Job is missing its upload path")
        return
    path = Path(job["path"])
    try:
        chunks, ids = index_file(path)
    except Exception as exc:
        logger.exception("Ingestion job failed job_id=%s file=%s", job["job_id"], job["file"])
        mark_failed(job["job_id"], str(exc) or "Indexing failed")
    else:
        mark_done(job["job_id"], chunks, len(ids))
        logger.info(
            "Ingestion job finished job_id=%s file=%s chunks=%d",
            job["job_id"],
            job["file"],
            chunks,
        )
    # Keep the upload if a status write failed, so the requeued job can retry it.
    path.unlink(missing_ok=True)


def main() -> None:
    configure_logging()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    logger.info("Ingestion worker started")
    backoff = 1
    while not _stopping:
        try:
            job = claim_job()
            if job:
                process_job(job)
        except RedisError:
            logger.warning("Redis unavailable; retrying in %ds", backoff, exc_info=True)
            _wait(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            continue
        backoff = 1
    logger.info("Ingestion worker stopped")


if __name__ == "__main__":
    main()
