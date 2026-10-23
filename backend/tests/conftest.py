from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

API_KEY = "test-key-123"


@pytest.fixture
def api_key() -> str:
    return API_KEY


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    # newline="\n" keeps LF endings on Windows so diff counts are identical everywhere.
    (root / "README.md").write_text("# Demo\nhello world\n", encoding="utf-8", newline="\n")
    (root / "src" / "app.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8", newline="\n"
    )
    (root / ".env").write_text("SECRET=topsecret\n", encoding="utf-8", newline="\n")
    return root


@pytest.fixture
def settings(workspace_dir: Path) -> Settings:
    return Settings(
        _env_file=None, app_env="test", api_key=API_KEY, workspace_root=str(workspace_dir)
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))
