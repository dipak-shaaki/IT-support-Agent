from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from redis.exceptions import RedisError
from uuid import UUID

from app.api import routes
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import AIMessage, HumanMessage


def create_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_health_returns_service_name():
    response = create_client().get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": routes.settings.app_name}


def test_root_health_mirrors_api_health():
    from app import main

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": routes.settings.app_name}


def test_get_conversation_returns_persisted_messages(monkeypatch):
    conversation_id = UUID("f0bb9e7a-7205-49c2-9daf-d4bcbe117c1a")
    monkeypatch.setattr(
        routes,
        "load_conversation",
        lambda requested_id, owner=None: [
            HumanMessage(content="How do I connect to VPN?"),
            AIMessage(content="Use the company VPN client."),
        ],
    )

    response = create_client().get(f"/api/conversations/{conversation_id}")

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "messages": [
            {"role": "user", "content": "How do I connect to VPN?"},
            {"role": "assistant", "content": "Use the company VPN client."},
        ],
    }


def test_chat_rejects_questions_shorter_than_two_characters():
    response = create_client().post("/api/chat", json={"question": "?"})

    assert response.status_code == 422


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
def test_chat_endpoints_reject_questions_over_token_limit(path):
    question = ("hello " * 401).strip()

    response = create_client().post(path, json={"question": question})

    assert response.status_code == 422
    assert "400-token limit" in response.text


def test_chat_returns_answer_and_writes_audit(monkeypatch):
    result = {
        "answer": "Use the VPN client.",
        "source_used": "private_kb",
        "trace": ["KB evidence grade → GOOD"],
        "citations": [{"title": "VPN guide", "url": "/vpn"}],
        "current_query": "connect to VPN",
    }
    audit_writes = []
    conversation_saves = []
    monkeypatch.setattr(routes, "load_conversation", lambda conversation_id, owner=None: [])
    monkeypatch.setattr(routes, "save_conversation", lambda conversation_id, messages, owner=None: conversation_saves.append((conversation_id, messages)))
    monkeypatch.setattr(
        routes, "ask", lambda question, history, request_id="": result
    )
    monkeypatch.setattr(
        routes,
        "write_audit",
        lambda question, source, trace, answer="": audit_writes.append(
            (question, source, trace, answer)
        ),
    )

    response = create_client().post(
        "/api/chat",
        json={"question": "How do I connect to VPN?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert UUID(body.pop("conversation_id"))
    assert body == {
        "answer": "Use the VPN client.",
        "source_used": "private_kb",
        "trace": ["KB evidence grade → GOOD"],
        "citations": [{"title": "VPN guide", "url": "/vpn"}],
        "retrieval_query": "connect to VPN",
    }
    assert audit_writes == [
        (
            "How do I connect to VPN?",
            "private_kb",
            ["KB evidence grade → GOOD"],
            "Use the VPN client.",
        )
    ]
    assert len(conversation_saves) == 1
    assert conversation_saves[0][1] == [
        HumanMessage(content="How do I connect to VPN?"),
        AIMessage(content="Use the VPN client."),
    ]


def test_chat_restores_history_for_follow_up(monkeypatch):
    conversation_id = UUID("f0bb9e7a-7205-49c2-9daf-d4bcbe117c1a")
    history = [
        HumanMessage(content="How do I connect to VPN?"),
        AIMessage(content="Use the company VPN client."),
    ]
    captured = {}
    monkeypatch.setattr(routes, "load_conversation", lambda requested_id, owner=None: history)
    monkeypatch.setattr(
        routes,
        "ask",
        lambda question, previous, request_id="": (
            captured.update(question=question, history=previous)
            or {
                "answer": "On macOS, install the company VPN client for Mac.",
                "source_used": "private_kb",
                "trace": [],
                "citations": [],
                "current_query": question,
            }
        ),
    )
    monkeypatch.setattr(routes, "write_audit", lambda *args: None)
    monkeypatch.setattr(routes, "save_conversation", lambda *args: None)

    response = create_client().post(
        "/api/chat",
        json={"question": "What about on Mac?", "conversation_id": str(conversation_id)},
    )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == str(conversation_id)
    assert captured == {"question": "What about on Mac?", "history": history}


def test_stream_returns_conversation_id_and_saves_turn(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="app.api.routes")
    conversation_id = UUID("f0bb9e7a-7205-49c2-9daf-d4bcbe117c1a")
    saved = []

    class FakeGraph:
        def stream(self, state, stream_mode):
            assert stream_mode == ["updates", "custom"]
            assert state["messages"] == [HumanMessage(content="Hello there")]
            assert state["stream_tokens"] is True
            yield "custom", {"type": "token", "text": "Hello! "}
            yield "custom", {"type": "token", "text": "How can I help?"}
            yield (
                "updates",
                {
                    "finalize": {
                        "answer": "Hello! How can I help?",
                        "source_used": "direct",
                        "trace": ["Direct response → no retrieval"],
                        "citations": [],
                    }
                }
            )

    monkeypatch.setattr(routes, "load_conversation", lambda requested_id, owner=None: [])
    monkeypatch.setattr(routes, "save_conversation", lambda key, messages, owner=None: saved.append((key, messages)))
    monkeypatch.setattr(routes, "write_audit", lambda *args: None)
    monkeypatch.setattr(routes, "agent_graph", FakeGraph())

    response = create_client().post(
        "/api/chat/stream",
        json={"question": "Hello there", "conversation_id": str(conversation_id)},
    )

    assert response.status_code == 200
    assert '"type": "token", "text": "Hello! "' in response.text
    assert '"type": "token", "text": "How can I help?"' in response.text
    assert f'"conversation_id": "{conversation_id}"' in response.text
    assert '"type": "done"' in response.text
    assert saved == [
        (
            conversation_id,
            [
                HumanMessage(content="Hello there"),
                AIMessage(content="Hello! How can I help?"),
            ],
        )
    ]
    assert "Chat stream started" in caplog.text
    assert "question_tokens=2" in caplog.text
    assert "Chat stream completed" in caplog.text


def test_stream_without_answer_saves_only_the_user_message(monkeypatch):
    conversation_id = UUID("f0bb9e7a-7205-49c2-9daf-d4bcbe117c1a")
    history = [
        HumanMessage(content="yes"),
        AIMessage(content="Can you share the error?"),
        HumanMessage(content="okay"),
        AIMessage(content="What does the message say?"),
        HumanMessage(content="yes"),
        AIMessage(content="Anything else you can share?"),
    ]
    saved = []

    class FakeGraph:
        def stream(self, state, stream_mode):
            yield (
                "updates",
                {"followup_guard": {"answer": "", "source_used": "direct"}},
            )

    monkeypatch.setattr(routes, "load_conversation", lambda requested_id, owner=None: history)
    monkeypatch.setattr(
        routes,
        "save_conversation",
        lambda key, messages, owner=None: saved.append((key, messages)),
    )
    monkeypatch.setattr(routes, "write_audit", lambda *args: None)
    monkeypatch.setattr(routes, "agent_graph", FakeGraph())

    response = create_client().post(
        "/api/chat/stream",
        json={"question": "yeah", "conversation_id": str(conversation_id)},
    )

    assert response.status_code == 200
    assert '"type": "done"' in response.text
    assert '"answer": ""' in response.text
    assert saved == [
        (conversation_id, [*history, HumanMessage(content="yeah")])
    ]


def test_stats_requires_admin_key(monkeypatch):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")

    response = create_client().get("/api/stats")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid admin key"}


def test_stats_returns_audit_summaries_with_admin_key(monkeypatch):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(routes, "source_breakdown", lambda: [{"source": "private_kb", "count": 2}])
    monkeypatch.setattr(routes, "daily_counts", lambda: [{"day": "2026-10-29", "count": 2}])
    monkeypatch.setattr(routes, "kb_gaps", lambda: [])
    monkeypatch.setattr(routes, "gap_total", lambda: 3)
    monkeypatch.setattr(routes, "recent_queries", lambda: [])

    response = create_client().get("/api/stats", headers={"x-admin-key": "test-secret"})

    assert response.status_code == 200
    assert response.json() == {
        "total": 2,
        "sources": [{"source": "private_kb", "count": 2}],
        "daily": [{"day": "2026-10-29", "count": 2}],
        "gaps": [],
        "gaps_total": 3,
        "recent": [],
    }


def test_ingestion_is_hidden_in_production_by_default(monkeypatch):
    from app.api import deps

    monkeypatch.setattr(deps.settings, "app_env", "production")
    monkeypatch.setattr(deps.settings, "allow_ingestion", None)
    client = create_client()

    upload = client.post(
        "/api/ingest",
        files={"file": ("vpn.md", b"VPN guidance", "text/markdown")},
    )
    job = client.get(f"/api/ingest/jobs/{'a' * 32}")

    assert upload.status_code == 404
    assert job.status_code == 404


def test_ingestion_can_be_enabled_in_production(monkeypatch):
    from app.api import deps

    monkeypatch.setattr(deps.settings, "app_env", "production")
    monkeypatch.setattr(deps.settings, "allow_ingestion", True)

    response = create_client().post(
        "/api/ingest",
        files={"file": ("vpn.md", b"VPN guidance", "text/markdown")},
    )

    assert response.status_code == 401


def test_ingest_requires_admin_key(monkeypatch):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")

    response = create_client().post(
        "/api/ingest",
        files={"file": ("vpn.md", b"VPN steps", "text/markdown")},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid admin key"}


def test_ingest_rejects_unsupported_file_type(monkeypatch):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")

    response = create_client().post(
        "/api/ingest",
        headers={"x-admin-key": "test-secret"},
        files={"file": ("policy.csv", b"policy", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Supported: .docx, .md, .pdf, .txt"}


def test_ingest_saves_parses_chunks_and_indexes_document(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(routes.settings, "upload_dir", str(tmp_path))
    parsed_files = []

    def fake_index_file(path: Path, original_filename: str):
        parsed_files.append((path.name, path.read_bytes(), original_filename))
        return 1, ["vector-1"]

    monkeypatch.setattr(routes, "index_file", fake_index_file)

    response = create_client().post(
        "/api/ingest",
        headers={"x-admin-key": "test-secret"},
        files={"file": ("vpn.md", b"VPN connection instructions", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "Document indexed",
        "file": "vpn.md",
        "chunks": 1,
        "ids_created": 1,
    }
    assert len(parsed_files) == 1
    assert parsed_files[0][0].endswith(".md")
    assert parsed_files[0][0] != "vpn.md"
    assert parsed_files[0][1] == b"VPN connection instructions"
    assert parsed_files[0][2] == "vpn.md"
    assert list(tmp_path.iterdir()) == []


def test_ingest_rejects_documents_without_indexable_text(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(routes.settings, "upload_dir", str(tmp_path))

    def empty_document(path, original_filename):
        raise routes.EmptyDocumentError("The uploaded document contains no indexable text")

    monkeypatch.setattr(routes, "index_file", empty_document)

    response = create_client().post(
        "/api/ingest",
        headers={"x-admin-key": "test-secret"},
        files={"file": ("vpn.md", b"", "text/markdown")},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "The uploaded document contains no indexable text"
    }
    assert list(tmp_path.iterdir()) == []


def test_ingest_rejects_oversized_upload_and_removes_partial_file(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(routes.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(routes.settings, "max_upload_bytes", 4)

    response = create_client().post(
        "/api/ingest",
        headers={"x-admin-key": "test-secret"},
        files={"file": ("vpn.md", b"12345", "text/markdown")},
    )

    assert response.status_code == 413
    assert "upload limit" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_ingest_queue_mode_returns_job_and_keeps_file_for_the_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(routes.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(routes.settings, "ingest_mode", "queue")
    created = []

    def fake_create_job(filename, path):
        created.append((filename, path))
        return "a" * 32

    monkeypatch.setattr(routes, "create_job", fake_create_job)

    response = create_client().post(
        "/api/ingest",
        headers={"x-admin-key": "test-secret"},
        files={"file": ("vpn.md", b"VPN connection instructions", "text/markdown")},
    )

    assert response.status_code == 202
    assert response.json() == {
        "job_id": "a" * 32,
        "status": "queued",
        "file": "vpn.md",
    }
    assert len(created) == 1
    assert created[0][0] == "vpn.md"
    assert Path(created[0][1]).parent == tmp_path
    assert Path(created[0][1]).name.endswith(".md")
    assert Path(created[0][1]).name != "vpn.md"
    assert Path(created[0][1]).read_bytes() == b"VPN connection instructions"


@pytest.mark.parametrize(
    "filename", ["../../outside.md", r"..\\..\\outside.md", "/tmp/outside.md"]
)
def test_ingest_uses_generated_path_for_traversal_filenames(monkeypatch, tmp_path, filename):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(routes.settings, "upload_dir", str(tmp_path / "uploads"))
    indexed = []

    def fake_index_file(path: Path, original_filename: str):
        indexed.append((path, original_filename))
        return 1, ["vector-1"]

    monkeypatch.setattr(routes, "index_file", fake_index_file)
    response = create_client().post(
        "/api/ingest",
        headers={"x-admin-key": "test-secret"},
        files={"file": (filename, b"valid text", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["file"] == "outside.md"
    assert indexed[0][0].parent == tmp_path / "uploads"
    assert indexed[0][0].name != "outside.md"
    assert indexed[0][1] == "outside.md"
    assert not (tmp_path / "outside.md").exists()
    assert list((tmp_path / "uploads").iterdir()) == []


def test_ingest_rejects_content_that_does_not_match_extension(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(routes.settings, "upload_dir", str(tmp_path))

    response = create_client().post(
        "/api/ingest",
        headers={"x-admin-key": "test-secret"},
        files={"file": ("not-a-pdf.pdf", b"plain text", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "File content does not match its extension"}
    assert list(tmp_path.iterdir()) == []


def test_ingest_rejects_symlink_escape_and_preserves_target(monkeypatch, tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("keep this file", encoding="utf-8")
    (upload_dir / f"{'a' * 32}.md").symlink_to(outside)
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(routes.settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(routes, "uuid4", lambda: SimpleNamespace(hex="a" * 32))

    response = create_client().post(
        "/api/ingest",
        headers={"x-admin-key": "test-secret"},
        files={"file": ("vpn.md", b"new content", "text/markdown")},
    )

    assert response.status_code == 400
    assert outside.read_text(encoding="utf-8") == "keep this file"


def test_concurrent_uploads_use_distinct_storage_names(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(routes.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(routes.settings, "ingest_mode", "queue")
    created = []

    def create_job(filename, path):
        created.append(Path(path))
        return "b" * 32

    monkeypatch.setattr(routes, "create_job", create_job)

    def upload(body: bytes):
        return create_client().post(
            "/api/ingest",
            headers={"x-admin-key": "test-secret"},
            files={"file": ("same.md", body, "text/markdown")},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(upload, [b"first document", b"second document"]))

    assert [response.status_code for response in responses] == [202, 202]
    assert len({path.name for path in created}) == 2
    assert {path.read_bytes() for path in created} == {b"first document", b"second document"}


def test_ingest_queue_failure_removes_the_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(routes.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(routes.settings, "ingest_mode", "queue")

    def broken_queue(filename, path):
        raise RedisError("down")

    monkeypatch.setattr(routes, "create_job", broken_queue)

    response = create_client().post(
        "/api/ingest",
        headers={"x-admin-key": "test-secret"},
        files={"file": ("vpn.md", b"VPN connection instructions", "text/markdown")},
    )

    assert response.status_code == 503
    assert list(tmp_path.iterdir()) == []


def test_ingest_job_returns_status_for_admins(monkeypatch):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    job = {
        "job_id": "a" * 32,
        "status": "done",
        "file": "vpn.md",
        "chunks": 3,
        "ids_created": 3,
        "error": "",
    }
    monkeypatch.setattr(routes, "get_job", lambda job_id: job)

    response = create_client().get(
        f"/api/ingest/jobs/{'a' * 32}", headers={"x-admin-key": "test-secret"}
    )

    assert response.status_code == 200
    assert response.json() == job


def test_ingest_job_requires_admin_key(monkeypatch):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")

    response = create_client().get(f"/api/ingest/jobs/{'a' * 32}")

    assert response.status_code == 401


def test_ingest_job_returns_404_for_unknown_job(monkeypatch):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(routes, "get_job", lambda job_id: None)

    response = create_client().get(
        f"/api/ingest/jobs/{'a' * 32}", headers={"x-admin-key": "test-secret"}
    )

    assert response.status_code == 404


def test_ready_reports_healthy_dependencies(monkeypatch):
    class FakeRedis:
        def ping(self):
            return True

    monkeypatch.setattr(routes, "get_redis_client", lambda: FakeRedis())
    monkeypatch.setattr(routes, "get_bm25_index", lambda: [object()])

    response = create_client().get("/api/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"redis": "ok", "bm25": "ok"},
    }


def test_ready_returns_503_when_redis_is_down(monkeypatch):
    class BrokenRedis:
        def ping(self):
            raise RuntimeError("down")

    monkeypatch.setattr(routes, "get_redis_client", lambda: BrokenRedis())
    monkeypatch.setattr(routes, "get_bm25_index", lambda: [])

    response = create_client().get("/api/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {"redis": "unavailable", "bm25": "empty"}
