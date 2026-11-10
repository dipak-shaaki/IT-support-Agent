from fastapi.testclient import TestClient

from app import main


def test_metrics_endpoint_exposes_http_request_counters():
    client = TestClient(main.app)

    client.get("/api/health")
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "it_support_http_requests_total" in response.text
    assert 'path="/api/health"' in response.text
