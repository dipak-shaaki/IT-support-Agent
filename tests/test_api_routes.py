from fastapi import FastAPI
from fastapi.testclient import TestClient
from uuid import UUID

from app.api import routes
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage


def create_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_health_returns_service_name():
    response = create_client().get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": routes.settings.app_name}


def test_get_conversation_returns_persisted_messages(monkeypatch):
    conversation_id = UUID("f0bb9e7a-7205-49c2-9daf-d4bcbe117c1a")
    monkeypatch.setattr(
        routes,
        "load_conversation",
        lambda requested_id: [
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
    monkeypatch.setattr(routes, "load_conversation", lambda conversation_id: [])
    monkeypatch.setattr(routes, "save_conversation", lambda conversation_id, messages: conversation_saves.append((conversation_id, messages)))
    monkeypatch.setattr(routes, "ask", lambda question, history: result)
    monkeypatch.setattr(
        routes,
        "write_audit",
        lambda question, source, trace: audit_writes.append((question, source, trace)),
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
        "rewritten_query": "connect to VPN",
    }
    assert audit_writes == [
        ("How do I connect to VPN?", "private_kb", ["KB evidence grade → GOOD"])
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
    monkeypatch.setattr(routes, "load_conversation", lambda requested_id: history)
    monkeypatch.setattr(
        routes,
        "ask",
        lambda question, previous: (
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


def test_stream_returns_conversation_id_and_saves_turn(monkeypatch):
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

    monkeypatch.setattr(routes, "load_conversation", lambda requested_id: [])
    monkeypatch.setattr(routes, "save_conversation", lambda key, messages: saved.append((key, messages)))
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
    monkeypatch.setattr(routes, "recent_queries", lambda: [])

    response = create_client().get("/api/stats", headers={"x-admin-key": "test-secret"})

    assert response.status_code == 200
    assert response.json() == {
        "total": 2,
        "sources": [{"source": "private_kb", "count": 2}],
        "daily": [{"day": "2026-10-29", "count": 2}],
        "gaps": [],
        "recent": [],
    }


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
    indexed_chunks = []

    def fake_load_file(path: Path):
        parsed_files.append((path.name, path.read_bytes()))
        return ["parsed document"]

    monkeypatch.setattr(routes, "load_file", fake_load_file)
    monkeypatch.setattr(
        routes,
        "chunk_documents",
        lambda documents: [f"{documents[0]} chunk"],
    )
    monkeypatch.setattr(
        routes,
        "replace_documents",
        lambda chunks: indexed_chunks.extend(chunks) or ["vector-1"],
    )

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
    assert parsed_files == [("vpn.md", b"VPN connection instructions")]
    assert indexed_chunks == ["parsed document chunk"]


def test_ingest_rejects_documents_without_indexable_text(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(routes.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(routes, "load_file", lambda path: [])
    monkeypatch.setattr(routes, "chunk_documents", lambda documents: [])

    def unexpected_replace(chunks):
        raise AssertionError("Empty documents must not replace indexed content")

    monkeypatch.setattr(routes, "replace_documents", unexpected_replace)

    response = create_client().post(
        "/api/ingest",
        headers={"x-admin-key": "test-secret"},
        files={"file": ("vpn.md", b"", "text/markdown")},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "The uploaded document contains no indexable text"
    }
