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


def test_frontend_routes_report_missing_production_build(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "FRONTEND_INDEX", tmp_path / "missing" / "index.html")
    response = TestClient(main.app).get("/")

    assert response.status_code == 503
    assert "frontend build is missing" in response.json()["detail"]
