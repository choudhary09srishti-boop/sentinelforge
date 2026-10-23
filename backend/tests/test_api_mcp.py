import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_repo_routes_require_api_key(client: TestClient) -> None:
    assert client.get("/api/v1/repo/files").status_code == 401


def test_listing_goes_through_the_mcp_gateway(settings: Settings, api_key: str) -> None:
    app = create_app(settings)
    client = TestClient(app)
    response = client.get("/api/v1/repo/files", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    assert "README.md" in {e["path"] for e in response.json()["entries"]}
    assert app.state.gateway.call_count == 1


def test_read_and_search_routes(client: TestClient, api_key: str) -> None:
    headers = {"X-API-Key": api_key}
    file = client.get("/api/v1/repo/file", params={"path": "src/app.py"}, headers=headers)
    assert file.status_code == 200
    assert "return a - b" in file.json()["content"]
    found = client.get("/api/v1/repo/search", params={"q": "hello"}, headers=headers)
    assert found.json()["matches"][0]["path"] == "README.md"


def test_summary_route(client: TestClient, api_key: str) -> None:
    response = client.get("/api/v1/repo/summary", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    assert response.json()["file_count"] == 2


def test_traversal_is_rejected_with_consistent_error(client: TestClient, api_key: str) -> None:
    response = client.get(
        "/api/v1/repo/file", params={"path": "../../etc/passwd"}, headers={"X-API-Key": api_key}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "http_400"


def test_sensitive_and_missing_files_map_to_403_and_404(client: TestClient, api_key: str) -> None:
    headers = {"X-API-Key": api_key}
    assert (
        client.get("/api/v1/repo/file", params={"path": ".env"}, headers=headers).status_code == 403
    )
    assert (
        client.get("/api/v1/repo/file", params={"path": "x.py"}, headers=headers).status_code == 404
    )


def test_propose_patch_route_returns_diff_and_leaves_file_untouched(
    client: TestClient, api_key: str, workspace_dir: Path
) -> None:
    target = workspace_dir / "src" / "app.py"
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    response = client.post(
        "/api/v1/repo/patches/propose",
        json={"path": "src/app.py", "new_content": "def add(a, b):\n    return a + b\n"},
        headers={"X-API-Key": api_key},
    )
    body = response.json()
    assert response.status_code == 200
    assert "+    return a + b" in body["diff"]
    assert body["applied"] is False and body["requires_approval"] is True
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before


def test_patch_request_validation_error_is_consistent(client: TestClient, api_key: str) -> None:
    response = client.post(
        "/api/v1/repo/patches/propose", json={"path": ""}, headers={"X-API-Key": api_key}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
