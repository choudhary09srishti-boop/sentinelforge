import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_app


def test_config_requires_api_key(client: TestClient) -> None:
    response = client.get("/api/v1/config")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "http_401"


def test_config_rejects_wrong_api_key(client: TestClient) -> None:
    response = client.get("/api/v1/config", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_config_accepts_valid_key_without_leaking_it(client: TestClient, api_key: str) -> None:
    response = client.get("/api/v1/config", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    assert response.json()["provider"]["mode"] == "demo"
    assert api_key not in response.text


def test_health_does_not_expose_secrets(client: TestClient, api_key: str) -> None:
    assert api_key not in client.get("/health").text


def test_unhandled_error_hides_internal_details(settings: Settings) -> None:
    app = create_app(settings)

    def boom() -> None:
        raise RuntimeError("secret internal detail")

    app.add_api_route("/boom", boom)
    response = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret internal detail" not in response.text


def test_production_rejects_default_api_key() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="production", api_key="change-me")


def test_cors_does_not_allow_unlisted_origin(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers
