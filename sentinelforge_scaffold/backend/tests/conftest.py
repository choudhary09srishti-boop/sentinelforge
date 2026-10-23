import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

API_KEY = "test-key-123"


@pytest.fixture
def api_key() -> str:
    return API_KEY


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, app_env="test", api_key=API_KEY)


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))
