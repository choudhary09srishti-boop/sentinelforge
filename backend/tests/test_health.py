from fastapi.testclient import TestClient


def test_health_reports_ok_and_provider_mode(client: TestClient) -> None:
    response = client.get("/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["provider"]["mode"] == "demo"
    assert body["provider"]["external"] is False
    assert response.headers["X-Request-ID"]


def test_unknown_route_uses_consistent_error_shape(client: TestClient) -> None:
    response = client.get("/does-not-exist")
    error = response.json()["error"]
    assert response.status_code == 404
    assert error["code"] == "http_404"
    assert error["request_id"]


def test_openapi_schema_is_exposed(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/health" in response.json()["paths"]
