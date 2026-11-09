import logging
import signal
from pathlib import Path

from app.core.logging import configure_logging
from app.services.ingest_jobs import claim_job, mark_done, mark_failed
from app.services.ingestion import index_file

logger = logging.getLogger(__name__)

_stopping = False


def _stop(_signum, _frame) -> None:
    global _stopping
    _stopping = True


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
    finally:
        # The vector and keyword indexes hold the content; the upload is no longer needed.
        path.unlink(missing_ok=True)


def main() -> None:
    configure_logging()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    logger.info("Ingestion worker started")
    while not _stopping:
        job = claim_job()
        if job:
            process_job(job)
    logger.info("Ingestion worker stopped")


if __name__ == "__main__":
    main()
