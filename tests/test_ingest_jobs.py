from app.services import ingest_jobs
from app import worker


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.expiries = {}
        self.queue = []

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
    monkeypatch.setattr(worker, "index_file", lambda path: (2, ["a", "b"]))
    monkeypatch.setattr(
        worker,
        "mark_done",
        lambda job_id, chunks, ids: calls.append((job_id, chunks, ids)),
    )

    worker.process_job({"job_id": "job-1", "file": "vpn.md", "path": str(upload)})

    assert calls == [("job-1", 2, 2)]
    assert not upload.exists()


def test_process_job_records_failure_and_removes_the_upload(monkeypatch, tmp_path):
    upload = tmp_path / "vpn.md"
    upload.write_text("")
    calls = []
    monkeypatch.setattr(
        worker,
        "index_file",
        lambda path: (_ for _ in ()).throw(ValueError("The uploaded document contains no indexable text")),
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
