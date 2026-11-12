import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.services import ingest_jobs
from app import worker


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.expiries = {}
        self.queue = []
        self.zsets = {}

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    def expire(self, key, seconds):
        self.expiries[key] = seconds

    def rpush(self, key, value):
        self.queue.append(value)

    def blpop(self, key, timeout=0):
        return (key, self.queue.pop(0)) if self.queue else None

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    def zrem(self, key, member):
        self.zsets.get(key, {}).pop(member, None)

    def zrangebyscore(self, key, minimum, maximum):
        low = float("-inf") if minimum == "-inf" else float(minimum)
        high = float("inf") if maximum == "+inf" else float(maximum)
        return [
            member
            for member, score in self.zsets.get(key, {}).items()
            if low <= score <= high
        ]


def test_claim_sets_a_lease_and_done_clears_it(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(ingest_jobs, "get_redis_client", lambda: client)
    job_id = ingest_jobs.create_job("vpn.md", "/uploads/vpn.md")

    ingest_jobs.claim_job(timeout=0)
    assert job_id in client.zsets[ingest_jobs.INGEST_LEASE_KEY]

    ingest_jobs.mark_done(job_id, 1, 1)
    assert job_id not in client.zsets[ingest_jobs.INGEST_LEASE_KEY]


def test_expired_lease_requeues_the_job(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(ingest_jobs, "get_redis_client", lambda: client)
    job_id = ingest_jobs.create_job("vpn.md", "/uploads/vpn.md")
    ingest_jobs.claim_job(timeout=0)

    requeued = ingest_jobs.requeue_expired_jobs(
        now=ingest_jobs.time.time() + 10_000
    )

    assert requeued == 1
    assert ingest_jobs.get_job(job_id)["status"] == "queued"
    assert client.queue == [job_id]


def test_stalled_job_fails_after_max_attempts_and_releases_the_upload(
    monkeypatch, tmp_path
):
    class Settings:
        ingest_job_ttl_seconds = 86400
        ingest_lease_seconds = 300
        ingest_max_attempts = 2

    client = FakeRedis()
    monkeypatch.setattr(ingest_jobs, "get_redis_client", lambda: client)
    monkeypatch.setattr(ingest_jobs, "get_settings", lambda: Settings())
    upload = tmp_path / "vpn.md"
    upload.write_text("VPN steps", encoding="utf-8")
    job_id = ingest_jobs.create_job("vpn.md", str(upload))

    ingest_jobs.claim_job(timeout=0)
    assert ingest_jobs.requeue_expired_jobs(now=ingest_jobs.time.time() + 10_000) == 1

    ingest_jobs.claim_job(timeout=0)
    assert ingest_jobs.requeue_expired_jobs(now=ingest_jobs.time.time() + 10_000) == 0

    assert ingest_jobs.get_job(job_id)["status"] == "error"
    assert not upload.exists()


def test_job_moves_from_queued_to_running_to_done(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(ingest_jobs, "get_redis_client", lambda: client)

    job_id = ingest_jobs.create_job("vpn.md", "/uploads/vpn.md")

    assert client.queue == [job_id]
    assert ingest_jobs.get_job(job_id) == {
        "job_id": job_id,
        "status": "queued",
        "file": "vpn.md",
        "chunks": 0,
        "ids_created": 0,
        "error": "",
    }
    assert ingest_jobs.claim_job(timeout=0) == {
        "job_id": job_id,
        "file": "vpn.md",
        "path": "/uploads/vpn.md",
    }
    assert ingest_jobs.get_job(job_id)["status"] == "running"

    ingest_jobs.mark_done(job_id, 2, 2)

    job = ingest_jobs.get_job(job_id)
    assert job["status"] == "done"
    assert job["chunks"] == 2
    assert job["ids_created"] == 2


def test_failed_job_records_a_bounded_error_and_ignores_bad_ids(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(ingest_jobs, "get_redis_client", lambda: client)
    job_id = ingest_jobs.create_job("vpn.md", "/uploads/vpn.md")

    ingest_jobs.mark_failed(job_id, "x" * 600)

    assert ingest_jobs.get_job(job_id)["status"] == "error"
    assert len(ingest_jobs.get_job(job_id)["error"]) == 500
    assert ingest_jobs.get_job("../../etc/passwd") is None


def test_process_job_marks_done_and_removes_the_upload(monkeypatch, tmp_path):
    upload = tmp_path / "vpn.md"
    upload.write_text("VPN steps")
    calls = []
    indexed = []
    monkeypatch.setattr(
        worker,
        "index_file",
        lambda path, original_filename: indexed.append((path, original_filename)) or (2, ["a", "b"]),
    )
    monkeypatch.setattr(
        worker,
        "mark_done",
        lambda job_id, chunks, ids: calls.append((job_id, chunks, ids)),
    )

    worker.process_job({"job_id": "job-1", "file": "vpn.md", "path": str(upload)})

    assert calls == [("job-1", 2, 2)]
    assert indexed == [(upload, "vpn.md")]
    assert not upload.exists()


def test_process_job_records_failure_and_removes_the_upload(monkeypatch, tmp_path):
    upload = tmp_path / "vpn.md"
    upload.write_text("")
    calls = []
    monkeypatch.setattr(
        worker,
        "index_file",
        lambda path, original_filename: (_ for _ in ()).throw(ValueError("The uploaded document contains no indexable text")),
    )
    monkeypatch.setattr(
        worker,
        "mark_failed",
        lambda job_id, error: calls.append((job_id, error)),
    )

    worker.process_job({"job_id": "job-1", "file": "vpn.md", "path": str(upload)})

    assert calls == [("job-1", "The uploaded document contains no indexable text")]
    assert not upload.exists()


def test_process_job_rejects_a_job_without_an_upload_path(monkeypatch):
    calls = []
    monkeypatch.setattr(
        worker,
        "mark_failed",
        lambda job_id, error: calls.append((job_id, error)),
    )

    worker.process_job({"job_id": "job-1", "file": "vpn.md", "path": ""})

    assert calls == [("job-1", "Job is missing its upload path")]


def test_worker_retries_after_a_redis_outage(monkeypatch):
    attempts = []

    def flaky_claim(timeout=5):
        attempts.append(1)
        if len(attempts) == 1:
            raise RedisConnectionError("redis is down")
        worker._stopping = True
        return None

    monkeypatch.setattr(worker, "claim_job", flaky_claim)
    monkeypatch.setattr(worker, "configure_logging", lambda: None)
    monkeypatch.setattr(worker, "_wait", lambda seconds: None)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    worker._stopping = False
    try:
        worker.main()
    finally:
        worker._stopping = False

    assert len(attempts) == 2


def test_process_job_keeps_the_upload_when_the_status_write_fails(
    monkeypatch, tmp_path
):
    upload = tmp_path / "vpn.md"
    upload.write_text("VPN steps")
    monkeypatch.setattr(worker, "index_file", lambda path, original_filename: (1, ["a"]))
    monkeypatch.setattr(
        worker,
        "mark_done",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RedisConnectionError("redis is down")
        ),
    )

    with pytest.raises(RedisConnectionError):
        worker.process_job(
            {"job_id": "job-1", "file": "vpn.md", "path": str(upload)}
        )

    assert upload.exists()
