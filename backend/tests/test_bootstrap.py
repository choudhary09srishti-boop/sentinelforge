from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.bootstrap import index_sample_docs, seed_workspace
from app.core.config import Settings
from app.core.domain import DomainError
from app.main import create_app
from app.rag.service import RagService

BUGGY = "def add(a, b):\n    return a - b\n"


def make_samples(tmp_path: Path) -> Path:
    samples = tmp_path / "samples"
    (samples / "workspace").mkdir(parents=True)
    (samples / "workspace" / "calculator.py").write_text(BUGGY, encoding="utf-8", newline="\n")
    (samples / "docs").mkdir()
    (samples / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    return samples


def demo_settings(tmp_path: Path, samples: Path, workspace: Path) -> Settings:
    return Settings(
        _env_file=None,
        api_key="test-key-123",
        workspace_root=str(workspace),
        data_dir=str(tmp_path / "data"),
        samples_dir=str(samples),
        bootstrap_samples=True,
    )


def test_seed_copies_and_a_second_seed_restores_the_original(tmp_path: Path) -> None:
    samples = make_samples(tmp_path)
    workspace = tmp_path / "ws"
    settings = demo_settings(tmp_path, samples, workspace)
    assert seed_workspace(settings) == 1
    target = workspace / "calculator.py"
    assert target.read_text(encoding="utf-8") == BUGGY
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    seed_workspace(settings)
    assert target.read_text(encoding="utf-8") == BUGGY


def test_seed_skips_caches_and_symlinks(tmp_path: Path) -> None:
    samples = make_samples(tmp_path)
    cache = samples / "workspace" / "__pycache__"
    cache.mkdir()
    (cache / "junk.pyc").write_bytes(b"x")
    try:
        (samples / "workspace" / "link.py").symlink_to(tmp_path / "outside.txt")
    except OSError:
        pytest.skip("symlinks are not available here")
    workspace = tmp_path / "ws"
    seed_workspace(demo_settings(tmp_path, samples, workspace))
    assert sorted(p.name for p in workspace.rglob("*") if p.is_file()) == ["calculator.py"]


def test_seed_never_copies_a_folder_onto_itself(tmp_path: Path) -> None:
    samples = make_samples(tmp_path)
    settings = demo_settings(tmp_path, samples, samples / "workspace")
    assert seed_workspace(settings) == 0


class RecordingRag:
    def __init__(self, fail: bool = False) -> None:
        self.names: list[str] = []
        self.fail = fail

    def ingest(self, name: str, data: bytes) -> None:
        if self.fail:
            raise DomainError("parse_error", "bad file")
        self.names.append(name)


def test_sample_documents_are_ingested(tmp_path: Path) -> None:
    samples = make_samples(tmp_path)
    settings = demo_settings(tmp_path, samples, tmp_path / "ws")
    rag = RecordingRag()
    assert index_sample_docs(settings, rag) == (1, 0)  # type: ignore[arg-type]
    assert rag.names == ["guide.md"]


def test_a_failing_sample_is_counted_not_fatal(tmp_path: Path) -> None:
    samples = make_samples(tmp_path)
    settings = demo_settings(tmp_path, samples, tmp_path / "ws")
    assert index_sample_docs(settings, RecordingRag(fail=True)) == (0, 1)  # type: ignore[arg-type]


def test_reset_is_hidden_unless_demo_mode_is_on(client: TestClient, api_key: str) -> None:
    response = client.post("/api/v1/demo/reset", headers={"X-API-Key": api_key})
    assert response.status_code == 404


def test_reset_requires_the_api_key(client: TestClient) -> None:
    assert client.post("/api/v1/demo/reset").status_code == 401


def test_reset_restores_the_sample_workspace(tmp_path: Path, rag_service: RagService) -> None:
    samples = make_samples(tmp_path)
    workspace = tmp_path / "ws"
    settings = demo_settings(tmp_path, samples, workspace)
    client = TestClient(create_app(settings, rag=rag_service))
    target = workspace / "calculator.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    response = client.post("/api/v1/demo/reset", headers={"X-API-Key": "test-key-123"})
    assert response.status_code == 200
    assert response.json()["files_restored"] == 1
    assert target.read_text(encoding="utf-8") == BUGGY
