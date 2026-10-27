"""Run the coding agent as a background job, then let a human apply or reject its patch.

Every run is persisted before it starts, so a restart leaves a readable record instead of a
silently lost task. Only `approve` writes to the workspace, and only from the patch that was
stored with the run: the request body can never supply file content.
"""

import asyncio
import json
import logging
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.agent.apply import apply_approved_patch
from app.agent.graph import SandboxPort, ToolCaller, build_graph
from app.agent.state import new_state
from app.api.tool_call import STATUS_BY_CODE
from app.core.config import Settings
from app.mcp_client import McpToolError
from app.mcp_server.workspace import Workspace, WorkspaceError
from app.providers.base import LLMProvider, ProviderError
from app.providers.factory import get_provider
from app.sandbox.runner import SandboxRunner
from app.storage.runs import RunRecord, RunStore

logger = logging.getLogger("sentinelforge")

LogFn = Callable[[str, str], None]
GraphRunner = Callable[[str, LogFn], Awaitable[dict[str, Any]]]

APPROVABLE = "awaiting_approval"
NOT_AWAITING = "Run is not awaiting approval."
NOT_FOUND = "Run not found."
RUNNING_STATUSES = {"queued", "running"}


def _node_message(node: str, state: dict[str, Any]) -> str:
    """One short, content-free line per finished node."""
    if node == "execute":
        return f"execute finished (tests: {state.get('test_status', 'not_run')})"
    if node == "review":
        return (
            f"review finished (approved: {bool(state.get('review_ok'))}, "
            f"attempts: {state.get('attempts', 0)})"
        )
    return f"{node} finished"


class AgentService:
    """Owns the lifecycle of one run: execute, then apply or reject on human decision."""

    def __init__(
        self,
        settings: Settings,
        store: RunStore,
        gateway: ToolCaller,
        *,
        workspace: Workspace | None = None,
        provider: LLMProvider | None = None,
        sandbox: SandboxPort | None = None,
        graph_runner: GraphRunner | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._gateway = gateway
        self._workspace = workspace
        self._provider = provider
        self._sandbox = sandbox
        self._graph_runner = graph_runner
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    @property
    def store(self) -> RunStore:
        return self._store

    # ---- execution ----------------------------------------------------------------------

    async def execute_run(self, run_id: str) -> None:
        """Background entry point. Never raises: failures are recorded on the run."""
        record = self._store.get_run(run_id)
        if record is None or not self._store.transition(run_id, {"queued"}, "running"):
            return
        self._log(run_id, "info", "Run started")
        try:
            state = await asyncio.wait_for(
                self._run_graph(run_id, record.task),
                timeout=self._settings.run_timeout_seconds,
            )
            await self._complete(run_id, state)
        except TimeoutError:
            self._fail(run_id, "Run timed out.")
        except ProviderError as exc:
            self._fail(run_id, str(exc))
        except Exception:
            logger.exception("Agent run %s failed unexpectedly", run_id)
            self._fail(run_id, "The run failed unexpectedly.")

    async def _run_graph(self, run_id: str, task: str) -> dict[str, Any]:
        def log(level: str, message: str) -> None:
            self._log(run_id, level, message)

        if self._graph_runner is not None:
            return await self._graph_runner(task, log)
        provider = self._provider or get_provider(self._settings)
        sandbox: SandboxPort = self._sandbox or SandboxRunner(
            Path(self._settings.workspace_root),
            timeout_seconds=self._settings.sandbox_timeout_seconds,
        )
        graph = build_graph(self._gateway, provider, sandbox=sandbox)
        state: dict[str, Any] = dict(new_state(task))
        async for chunk in graph.astream(new_state(task), stream_mode="updates"):
            for node, update in chunk.items():
                if isinstance(update, dict):
                    state.update(update)
                log("info", _node_message(str(node), state))
        return state

    async def _complete(self, run_id: str, state: dict[str, Any]) -> None:
        """Persist the final state and either await approval or mark the run completed."""
        state_json = json.dumps(state, default=str)
        patch_path = str(state.get("patch_path") or "")
        patch_content = str(state.get("patch_content") or "")
        if not patch_path or not patch_content:
            if self._store.transition(run_id, {"running"}, "completed", state_json=state_json):
                self._log(run_id, "info", "Finished without a patch")
            return
        try:
            proposal = await self._gateway.call(
                "propose_patch", {"path": patch_path, "new_content": patch_content}
            )
        except McpToolError as exc:
            self._fail(run_id, exc.message, state_json=state_json)
            return
        patch_id = str(proposal.get("patch_id", ""))
        base_sha256 = str(proposal.get("base_sha256", ""))
        if patch_id != str(state.get("patch_id") or ""):
            self._fail(
                run_id,
                "The proposed patch no longer matches the workspace. Start a new run.",
                state_json=state_json,
            )
            return
        moved = self._store.transition(
            run_id,
            {"running"},
            "awaiting_approval",
            state_json=state_json,
            base_sha256=base_sha256,
        )
        if moved:
            self._log(run_id, "info", "Patch ready for approval")

    def _fail(self, run_id: str, message: str, **fields: str) -> None:
        self._store.transition(run_id, RUNNING_STATUSES, "failed", error=message, **fields)
        self._log(run_id, "error", message)

    def _log(self, run_id: str, level: str, message: str) -> None:
        self._store.add_event(run_id, level, message)

    # ---- human decision -----------------------------------------------------------------

    async def approve(self, run_id: str) -> RunRecord:
        """Apply the stored patch to the real workspace, then record the result."""
        record = self._store.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=NOT_FOUND)
        lock = self._lock_for(run_id)
        applied: RunRecord = await asyncio.to_thread(self._apply_locked, run_id, lock)
        return applied

    def _apply_locked(self, run_id: str, lock: threading.Lock) -> RunRecord:
        """The whole check-then-apply sequence runs under one per-run lock, off the loop."""
        with lock:
            record = self._store.get_run(run_id)
            if record is None:
                raise HTTPException(status_code=404, detail=NOT_FOUND)
            if record.status != APPROVABLE:
                raise HTTPException(status_code=409, detail=NOT_AWAITING)
            state = record.state
            path = str(state.get("patch_path") or "")
            content = str(state.get("patch_content") or "")
            patch_id = str(state.get("patch_id") or "")
            if not path or not content or not patch_id or not record.base_sha256:
                raise HTTPException(status_code=409, detail=NOT_AWAITING)
            try:
                result = apply_approved_patch(
                    self._resolve_workspace(),
                    path=path,
                    new_content=content,
                    base_sha256=record.base_sha256,
                    patch_id=patch_id,
                )
            except WorkspaceError as exc:
                code = str(getattr(exc, "code", "internal_error"))
                message = str(getattr(exc, "message", "Request failed."))
                if code == "conflict":
                    self._store.transition(run_id, {APPROVABLE}, "failed", error=message)
                self._log(run_id, "error", message)
                status = 409 if code == "conflict" else STATUS_BY_CODE.get(code, 500)
                raise HTTPException(status_code=status, detail=message) from None
            apply_json = json.dumps(result.model_dump())
            if not self._store.transition(run_id, {APPROVABLE}, "applied", apply_json=apply_json):
                raise HTTPException(status_code=409, detail=NOT_AWAITING)
            self._log(run_id, "info", f"Patch applied to {result.path}")
            stored = self._store.get_run(run_id)
            if stored is None:
                raise HTTPException(status_code=404, detail=NOT_FOUND)
            return stored

    def reject(self, run_id: str, reason: str) -> RunRecord:
        """Record a human rejection. Nothing is ever written for a rejected run."""
        with self._lock_for(run_id):
            record = self._store.get_run(run_id)
            if record is None:
                raise HTTPException(status_code=404, detail=NOT_FOUND)
            moved = self._store.transition(
                run_id, {APPROVABLE}, "rejected", reject_reason=reason[:500]
            )
            if not moved:
                raise HTTPException(status_code=409, detail=NOT_AWAITING)
            self._log(run_id, "info", "Patch rejected by a human reviewer")
            stored = self._store.get_run(run_id)
            if stored is None:
                raise HTTPException(status_code=404, detail=NOT_FOUND)
            return stored

    # ---- helpers ------------------------------------------------------------------------

    def _lock_for(self, run_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(run_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[run_id] = lock
            return lock

    def _resolve_workspace(self) -> Workspace:
        if self._workspace is not None:
            return self._workspace
        return Workspace(
            Path(self._settings.workspace_root), max_read_bytes=self._settings.max_read_bytes
        )
