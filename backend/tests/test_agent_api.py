"""Agent run API: lifecycle, logs, approval, rejection and failure containment.

Every test is offline: the graph is replaced by a deterministic runner, and the only real
moving parts are the SQLite run store, the MCP gateway and the workspace on disk.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agent.apply import apply_approved_patch
from app.agent.service import GraphRunner
from app.agent.state import new_state
from app.core.config import Settings
from app.main import create_app
from app.mcp_server.workspace import Workspace, WorkspaceError
from app.providers.base import ProviderError
from app.rag.service import RagService

TASK = "Fix add() so it adds the two numbers in src/app.py"
PATCH_CONTENT = "def add(a, b):\n    return a + b\n"
RUNS = "/api/v1/agent/runs"
UNKNOWN_ID = "0" * 16
ALL_ID = "a" * 16


def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key}


def _proposal(workspace_dir: Path) -> Any:
    return Workspace(workspace_dir).propose_patch("src/app.py", PATCH_CONTENT)


def make_runner(workspace_dir: Path, *, with_patch: bool = True) -> GraphRunner:
    """Deterministic stand-in for the LangGraph run, with a real patch id for the fixture."""
    proposal = _proposal(workspace_dir)

    async def runner(task: str, log: Callable[[str, str], None]) -> dict[str, Any]:
        log("info", "analyze finished")
        log("info", "patch finished")
        state: dict[str, Any] = dict(new_state(task))
        state.update(
            {
                "analysis": "add() subtracts instead of adding.",
                "search_terms": ["add"],
                "plan": ["Read src/app.py", "Return a + b"],
                "evidence": ["src/app.py:1"],
                "inspected_files": ["src/app.py"],
                "diff": proposal.diff if with_patch else "",
                "patch_path": "src/app.py" if with_patch else "",
                "patch_id": proposal.patch_id if with_patch else "",
                "patch_content": PATCH_CONTENT if with_patch else "",
                "test_status": "passed",
                "test_output": "1 passed",
                "risks": [],
                "attempts": 1,
                "review_ok": True,
                "report": "Fixed add().",
            }
        )
        return state

    return runner


def make_failing_runner(error: Exception) -> GraphRunner:
    async def runner(task: str, log: Callable[[str, str], None]) -> dict[str, Any]:
        log("info", "analyze finished")
        raise error

    return runner


def build_client(settings: Settings, rag: RagService, runner: GraphRunner) -> TestClient:
    return TestClient(create_app(settings, rag=rag, graph_runner=runner))


@pytest.fixture
def agent_client(settings: Settings, rag_service: RagService, workspace_dir: Path) -> TestClient:
    return build_client(settings, rag_service, make_runner(workspace_dir))


def start_run(client: TestClient, api_key: str, task: str = TASK) -> str:
    response = client.post(RUNS, json={"task": task}, headers=_headers(api_key))
    assert response.status_code == 202
    run_id: str = response.json()["id"]
    assert len(run_id) == 16
    return run_id


def get_run(client: TestClient, api_key: str, run_id: str) -> dict[str, Any]:
    response = client.get(f"{RUNS}/{run_id}", headers=_headers(api_key))
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


def get_logs(client: TestClient, api_key: str, run_id: str, query: str = "") -> dict[str, Any]:
    response = client.get(f"{RUNS}/{run_id}/logs{query}", headers=_headers(api_key))
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


def messages(logs: dict[str, Any]) -> list[str]:
    return [event["message"] for event in logs["events"]]


ROUTES = [
    ("POST", RUNS, {"json": {"task": TASK}}),
    ("GET", RUNS, {}),
    ("GET", f"{RUNS}/{ALL_ID}", {}),
    ("GET", f"{RUNS}/{ALL_ID}/logs", {}),
    ("POST", f"{RUNS}/{ALL_ID}/approve", {}),
    ("POST", f"{RUNS}/{ALL_ID}/reject", {}),
]


@pytest.mark.parametrize(("method", "path", "kwargs"), ROUTES)
def test_every_route_requires_the_api_key(
    agent_client: TestClient, method: str, path: str, kwargs: dict[str, Any]
) -> None:
    response = agent_client.request(method, path, **kwargs)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "http_401"


@pytest.mark.parametrize(("method", "path", "kwargs"), ROUTES)
def test_every_route_rejects_a_wrong_key(
    agent_client: TestClient, method: str, path: str, kwargs: dict[str, Any]
) -> None:
    response = agent_client.request(method, path, headers={"X-API-Key": "wrong"}, **kwargs)
    assert response.status_code == 401


def test_run_reaches_awaiting_approval(
    agent_client: TestClient, api_key: str, workspace_dir: Path
) -> None:
    run_id = start_run(agent_client, api_key)
    detail = get_run(agent_client, api_key, run_id)

    assert detail["status"] == "awaiting_approval"
    assert detail["has_patch"] is True
    assert detail["patch_path"] == "src/app.py"
    assert detail["test_status"] == "passed"
    assert detail["plan"] == ["Read src/app.py", "Return a + b"]
    assert detail["attempts"] == 1
    assert detail["review_ok"] is True
    assert detail["apply_result"] is None
    assert detail["error"] == ""
    assert "return a + b" in detail["diff"]

    log_messages = messages(get_logs(agent_client, api_key, run_id))
    assert "Run queued" in log_messages
    assert "analyze finished" in log_messages
    assert "patch finished" in log_messages
    assert "Patch ready for approval" in log_messages
    # The stub proposed nothing yet: only the approval API may write.
    assert (workspace_dir / "src" / "app.py").read_text(encoding="utf-8") != PATCH_CONTENT


def test_create_run_returns_202_and_a_queued_summary(
    agent_client: TestClient, api_key: str
) -> None:
    response = agent_client.post(RUNS, json={"task": TASK}, headers=_headers(api_key))
    assert response.status_code == 202
    body = response.json()
    assert body["id"] and body["task"] == TASK
    assert body["created_at"] and body["updated_at"]
    assert body["status"] in {"queued", "awaiting_approval"}


def test_run_list_contains_the_run(agent_client: TestClient, api_key: str) -> None:
    run_id = start_run(agent_client, api_key)
    response = agent_client.get(RUNS, headers=_headers(api_key))
    assert response.status_code == 200
    runs = response.json()["runs"]
    assert [run["id"] for run in runs] == [run_id]
    assert runs[0]["has_patch"] is True


def test_logs_paging_follows_next_after(agent_client: TestClient, api_key: str) -> None:
    run_id = start_run(agent_client, api_key)
    full = get_logs(agent_client, api_key, run_id)
    assert len(full["events"]) >= 3

    first = get_logs(agent_client, api_key, run_id, "?limit=1")
    assert len(first["events"]) == 1
    assert first["next_after"] == first["events"][0]["id"]

    second = get_logs(agent_client, api_key, run_id, f"?after={first['next_after']}&limit=1")
    assert len(second["events"]) == 1
    assert second["events"][0]["id"] > first["next_after"]

    tail_after = full["next_after"]
    empty = get_logs(agent_client, api_key, run_id, f"?after={tail_after}")
    assert empty["events"] == []
    assert empty["next_after"] == tail_after
    assert [event["id"] for event in full["events"]] == sorted(
        event["id"] for event in full["events"]
    )


def test_approve_applies_the_patch_once(
    agent_client: TestClient, api_key: str, workspace_dir: Path
) -> None:
    run_id = start_run(agent_client, api_key)

    response = agent_client.post(f"{RUNS}/{run_id}/approve", headers=_headers(api_key))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "applied"
    assert body["apply_result"]["applied"] is True
    assert body["apply_result"]["path"] == "src/app.py"
    assert len(body["apply_result"]["new_sha256"]) == 64
    assert (workspace_dir / "src" / "app.py").read_text(encoding="utf-8") == PATCH_CONTENT
    assert "Patch applied to src/app.py" in messages(get_logs(agent_client, api_key, run_id))

    again = agent_client.post(f"{RUNS}/{run_id}/approve", headers=_headers(api_key))
    assert again.status_code == 409
    assert again.json()["error"]["message"] == "Run is not awaiting approval."


def test_reject_never_touches_the_file(
    agent_client: TestClient, api_key: str, workspace_dir: Path
) -> None:
    target = workspace_dir / "src" / "app.py"
    before = target.read_bytes()
    run_id = start_run(agent_client, api_key)

    response = agent_client.post(
        f"{RUNS}/{run_id}/reject",
        json={"reason": "Wrong approach for this release."},
        headers=_headers(api_key),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["reject_reason"] == "Wrong approach for this release."
    assert target.read_bytes() == before

    approve = agent_client.post(f"{RUNS}/{run_id}/approve", headers=_headers(api_key))
    assert approve.status_code == 409
    assert target.read_bytes() == before


def test_reject_without_a_body_is_allowed(agent_client: TestClient, api_key: str) -> None:
    run_id = start_run(agent_client, api_key)
    response = agent_client.post(f"{RUNS}/{run_id}/reject", headers=_headers(api_key))
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_approve_fails_when_the_file_changed_after_the_run(
    agent_client: TestClient, api_key: str, workspace_dir: Path
) -> None:
    run_id = start_run(agent_client, api_key)
    target = workspace_dir / "src" / "app.py"
    target.write_text("def add(a, b):\n    return 0\n", encoding="utf-8", newline="\n")

    response = agent_client.post(f"{RUNS}/{run_id}/approve", headers=_headers(api_key))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "http_409"

    detail = get_run(agent_client, api_key, run_id)
    assert detail["status"] == "failed"
    assert detail["error"]
    assert "return 0" in target.read_text(encoding="utf-8")


def test_run_without_a_patch_completes(
    settings: Settings, rag_service: RagService, workspace_dir: Path, api_key: str
) -> None:
    client = build_client(settings, rag_service, make_runner(workspace_dir, with_patch=False))
    run_id = start_run(client, api_key)

    detail = get_run(client, api_key, run_id)
    assert detail["status"] == "completed"
    assert detail["has_patch"] is False
    assert "Finished without a patch" in messages(get_logs(client, api_key, run_id))

    approve = client.post(f"{RUNS}/{run_id}/approve", headers=_headers(api_key))
    assert approve.status_code == 409


def test_unexpected_error_is_hidden(
    settings: Settings, rag_service: RagService, api_key: str
) -> None:
    client = build_client(settings, rag_service, make_failing_runner(RuntimeError("secret-xyz")))
    run_id = start_run(client, api_key)

    detail_response = client.get(f"{RUNS}/{run_id}", headers=_headers(api_key))
    logs_response = client.get(f"{RUNS}/{run_id}/logs", headers=_headers(api_key))
    detail = detail_response.json()
    assert detail["status"] == "failed"
    assert detail["error"] == "The run failed unexpectedly."
    assert "secret-xyz" not in detail_response.text
    assert "secret-xyz" not in logs_response.text
    assert "Traceback" not in logs_response.text


def test_provider_error_message_is_shown(
    settings: Settings, rag_service: RagService, api_key: str
) -> None:
    client = build_client(
        settings, rag_service, make_failing_runner(ProviderError("bad key config"))
    )
    run_id = start_run(client, api_key)

    detail = get_run(client, api_key, run_id)
    assert detail["status"] == "failed"
    assert detail["error"] == "bad key config"


def test_unknown_run_returns_404(agent_client: TestClient, api_key: str) -> None:
    response = agent_client.get(f"{RUNS}/{UNKNOWN_ID}", headers=_headers(api_key))
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Run not found."

    approve = agent_client.post(f"{RUNS}/{UNKNOWN_ID}/approve", headers=_headers(api_key))
    assert approve.status_code == 404
    reject = agent_client.post(f"{RUNS}/{UNKNOWN_ID}/reject", headers=_headers(api_key))
    assert reject.status_code == 404
    logs = agent_client.get(f"{RUNS}/{UNKNOWN_ID}/logs", headers=_headers(api_key))
    assert logs.status_code == 404


def test_malformed_run_id_returns_422(agent_client: TestClient, api_key: str) -> None:
    response = agent_client.get(f"{RUNS}/ZZZ", headers=_headers(api_key))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_short_task_returns_422(agent_client: TestClient, api_key: str) -> None:
    for bad in ("tiny", "     a     "):
        response = agent_client.post(RUNS, json={"task": bad}, headers=_headers(api_key))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"


def test_too_many_active_runs_returns_429(
    settings: Settings, rag_service: RagService, workspace_dir: Path, api_key: str
) -> None:
    limited = settings.model_copy(update={"max_active_runs": 1})
    client = build_client(limited, rag_service, make_runner(workspace_dir))
    client.app.state.run_store.create_run("An already queued task")

    response = client.post(RUNS, json={"task": TASK}, headers=_headers(api_key))
    assert response.status_code == 429
    assert response.json()["error"]["message"] == "Too many active runs. Try again shortly."


def test_responses_never_contain_patch_internals(agent_client: TestClient, api_key: str) -> None:
    create = agent_client.post(RUNS, json={"task": TASK}, headers=_headers(api_key))
    run_id = create.json()["id"]
    detail = agent_client.get(f"{RUNS}/{run_id}", headers=_headers(api_key))
    approve = agent_client.post(f"{RUNS}/{run_id}/approve", headers=_headers(api_key))
    listing = agent_client.get(RUNS, headers=_headers(api_key))
    logs = agent_client.get(f"{RUNS}/{run_id}/logs", headers=_headers(api_key))

    for response in (create, detail, approve, listing, logs):
        assert response.status_code in {200, 202}
        assert "patch_content" not in response.text
        assert "base_sha256" not in response.text


def test_openapi_documents_the_agent_api(agent_client: TestClient) -> None:
    spec = agent_client.get("/openapi.json").json()
    paths = spec["paths"]
    assert f"{RUNS}" in paths
    assert f"{RUNS}/{{run_id}}" in paths
    assert f"{RUNS}/{{run_id}}/logs" in paths
    assert f"{RUNS}/{{run_id}}/approve" in paths
    assert f"{RUNS}/{{run_id}}/reject" in paths

    operations = [
        operation
        for path, item in paths.items()
        if path.startswith("/api/v1/agent")
        for method, operation in item.items()
        if method in {"get", "post"}
    ]
    assert len(operations) == 6
    for operation in operations:
        assert operation["summary"]
        assert operation["description"]

    detail_schema = spec["components"]["schemas"]["RunDetail"]
    assert "patch_content" not in detail_schema["properties"]
    assert "base_sha256" not in detail_schema["properties"]


def test_the_write_path_is_hash_bound(workspace_dir: Path) -> None:
    """Second line of defence behind the status guard: the same patch cannot be applied twice."""
    proposal = _proposal(workspace_dir)
    workspace = Workspace(workspace_dir)
    result = apply_approved_patch(
        workspace,
        path="src/app.py",
        new_content=PATCH_CONTENT,
        base_sha256=proposal.base_sha256,
        patch_id=proposal.patch_id,
    )
    assert result.applied is True
    assert (workspace_dir / "src" / "app.py").read_text(encoding="utf-8") == PATCH_CONTENT

    with pytest.raises(WorkspaceError):
        apply_approved_patch(
            workspace,
            path="src/app.py",
            new_content=PATCH_CONTENT,
            base_sha256=proposal.base_sha256,
            patch_id=proposal.patch_id,
        )
