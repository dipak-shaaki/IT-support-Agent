from fastapi.testclient import TestClient

from app import main


def test_frontend_routes_serve_react_entrypoint(monkeypatch, tmp_path):
    index = tmp_path / "index.html"
    index.write_text("<div id='root'>React app</div>", encoding="utf-8")
    monkeypatch.setattr(main, "FRONTEND_INDEX", index)
    client = TestClient(main.app)

    assert client.get("/").text == "<div id='root'>React app</div>"
    assert client.get("/dashboard").text == "<div id='root'>React app</div>"


def test_startup_preloads_embedding_model(monkeypatch):
    preloads = []
    monkeypatch.setattr(main, "get_embeddings", lambda: preloads.append(True))

    with TestClient(main.app):
        assert preloads == [True]


def test_api_allows_configured_frontend_origin():
    response = TestClient(main.app).options(
        "/api/chat/stream",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_frontend_routes_report_missing_production_build(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "FRONTEND_INDEX", tmp_path / "missing" / "index.html")
    response = TestClient(main.app).get("/")

    assert response.status_code == 503
    assert "frontend build is missing" in response.json()["detail"]


def test_request_id_is_generated_or_propagated():
    client = TestClient(main.app)

    generated = client.get("/api/health")
    propagated = client.get("/api/health", headers={"X-Request-ID": "abc-123"})

    assert generated.headers["X-Request-ID"]
    assert propagated.headers["X-Request-ID"] == "abc-123"
