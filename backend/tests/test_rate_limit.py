from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.rate_limit import RateLimitSettings
from app.main import create_app
from app.rag.service import RagService


def make_client(
    settings: Settings, rag: RagService, general: int, heavy: int = 50, trust_proxy: bool = False
) -> TestClient:
    limits = RateLimitSettings(
        _env_file=None,
        rate_limit_per_minute=general,
        rate_limit_heavy_per_minute=heavy,
        rate_limit_trust_proxy=trust_proxy,
    )
    return TestClient(create_app(settings, rag=rag, rate_limits=limits))


def test_health_is_never_limited(settings: Settings, rag_service: RagService) -> None:
    client = make_client(settings, rag_service, general=1)
    assert [client.get("/health").status_code for _ in range(5)] == [200] * 5


def test_requests_over_the_limit_get_429(
    settings: Settings, rag_service: RagService, api_key: str
) -> None:
    client = make_client(settings, rag_service, general=3)
    headers = {"X-API-Key": api_key}
    codes = [client.get("/api/v1/config", headers=headers).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    blocked = client.get("/api/v1/config", headers=headers)
    assert blocked.json()["error"]["code"] == "rate_limited"
    assert int(blocked.headers["Retry-After"]) >= 1


def test_unauthenticated_guessing_is_throttled(settings: Settings, rag_service: RagService) -> None:
    client = make_client(settings, rag_service, general=2)
    codes = [
        client.get("/api/v1/config", headers={"X-API-Key": f"guess-{i}"}).status_code
        for i in range(3)
    ]
    assert codes == [401, 401, 429]


def test_expensive_posts_have_a_smaller_limit(
    settings: Settings, rag_service: RagService, api_key: str
) -> None:
    client = make_client(settings, rag_service, general=100, heavy=2)
    headers = {"X-API-Key": api_key}
    codes = [
        client.post(
            "/api/v1/documents", headers=headers, files={"file": ("x.exe", b"data")}
        ).status_code
        for _ in range(3)
    ]
    assert codes[:2] == [415, 415]
    assert codes[2] == 429
    assert client.get("/api/v1/documents", headers=headers).status_code == 200


def test_spoofed_forwarding_header_is_ignored_by_default(
    settings: Settings, rag_service: RagService
) -> None:
    client = make_client(settings, rag_service, general=1)
    first = client.get("/api/v1/config", headers={"X-Forwarded-For": "1.1.1.1"})
    second = client.get("/api/v1/config", headers={"X-Forwarded-For": "2.2.2.2"})
    assert first.status_code == 401
    assert second.status_code == 429


def test_each_proxied_client_gets_its_own_bucket(
    settings: Settings, rag_service: RagService
) -> None:
    client = make_client(settings, rag_service, general=1, trust_proxy=True)
    a1 = client.get("/api/v1/config", headers={"X-Forwarded-For": "1.1.1.1"})
    a2 = client.get("/api/v1/config", headers={"X-Forwarded-For": "1.1.1.1"})
    b1 = client.get("/api/v1/config", headers={"X-Forwarded-For": "2.2.2.2"})
    assert (a1.status_code, a2.status_code, b1.status_code) == (401, 429, 401)
