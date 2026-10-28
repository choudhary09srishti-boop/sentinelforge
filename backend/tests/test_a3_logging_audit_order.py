import json
import logging

import pytest
from test_mcp_sandbox import FakeSandbox, gateway_for

from app.agent.graph import build_graph
from app.core.log import JsonFormatter
from app.mcp_client import McpToolError
from app.mcp_server.workspace import Workspace
from app.providers.demo import DemoProvider
from app.sandbox.runner import SandboxError, SandboxResult, _prepare_copy


def make_record(**fields: object) -> logging.LogRecord:
    record = logging.LogRecord("sentinelforge.test", logging.INFO, "f.py", 1, "hello", (), None)
    record.fields = fields
    return record


def passed() -> SandboxResult:
    return SandboxResult(status="passed", exit_code=0, output="1 passed", duration_seconds=0.2)


def make_workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "notes.txt").write_text("hi", encoding="utf-8")
    (root / "blob.bin").write_bytes(b"\x00\x01")
    (root / "tool.exe").write_bytes(b"MZ")
    return Workspace(root)


def test_log_lines_are_single_line_json() -> None:
    line = JsonFormatter().format(make_record(note="a\nb", n=3))
    assert "\n" not in line
    data = json.loads(line)
    assert data["note"] == "a\nb" and data["n"] == 3 and data["level"] == "info"


def test_long_values_are_clipped() -> None:
    data = json.loads(JsonFormatter().format(make_record(big="x" * 1000)))
    assert len(data["big"]) < 400


def test_fields_cannot_overwrite_base_keys() -> None:
    data = json.loads(JsonFormatter().format(make_record(level="forged", message="forged")))
    assert data["level"] == "info" and data["message"] == "hello"


def test_every_request_is_logged_with_its_request_id(client, caplog) -> None:
    caplog.set_level(logging.INFO, logger="sentinelforge")
    response = client.get("/health")
    records = [r for r in caplog.records if r.name == "sentinelforge.request"]
    assert records
    assert records[-1].fields["request_id"] == response.headers["X-Request-ID"]


async def test_sandbox_runs_leave_audit_events(settings, rag_service, caplog) -> None:
    caplog.set_level(logging.INFO, logger="sentinelforge")
    gateway = gateway_for(settings, rag_service, FakeSandbox(passed()))
    arguments = {"patch_path": "src/app.py", "patch_content": "x = 1\n"}
    await gateway.call("run_sandbox_tests", arguments)
    records = [r for r in caplog.records if r.name == "sentinelforge.audit"]
    assert [r.fields["event"] for r in records] == ["sandbox.run.start", "sandbox.run.finish"]
    assert records[-1].fields["status"] == "passed"
    assert records[-1].fields["patch_path"] == "src/app.py"


async def test_rejected_sandbox_calls_are_audited(settings, rag_service, caplog) -> None:
    caplog.set_level(logging.INFO, logger="sentinelforge")
    gateway = gateway_for(settings, rag_service, FakeSandbox(passed()))
    with pytest.raises(McpToolError):
        await gateway.call("run_sandbox_tests", {"patch_path": "src/app.py"})
    records = [r for r in caplog.records if r.name == "sentinelforge.audit"]
    assert [r.fields["event"] for r in records] == ["sandbox.run.rejected"]


def test_only_allowlisted_file_types_enter_the_sandbox(tmp_path) -> None:
    dest = tmp_path / "copy"
    dest.mkdir()
    _prepare_copy(make_workspace(tmp_path), dest, None, None)
    copied = sorted(p.name for p in dest.rglob("*") if p.is_file())
    assert copied == ["a.py", "notes.txt"]


def test_patching_a_disallowed_file_type_is_refused(tmp_path) -> None:
    dest = tmp_path / "copy"
    dest.mkdir()
    with pytest.raises(SandboxError):
        _prepare_copy(make_workspace(tmp_path), dest, "tool.exe", "boom")
    assert not (dest / "tool.exe").exists()


def test_agent_follows_the_required_step_order() -> None:
    graph = build_graph(None, DemoProvider())
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    required = [
        ("analyze", "plan"),
        ("plan", "retrieve"),
        ("retrieve", "inspect"),
        ("inspect", "patch"),
        ("patch", "execute"),
        ("execute", "review"),
        ("review", "report"),
    ]
    for pair in required:
        assert pair in edges
    assert ("analyze", "retrieve") not in edges
