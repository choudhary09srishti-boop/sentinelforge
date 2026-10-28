"""Agent run routes: start a run, poll it, read its log, approve or reject its patch.

The agent runs as a FastAPI background task, so no lifespan handler or task registry is
needed: the response is returned first, then execution continues in the same app.
"""

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query, Request

from app.agent.apply import ApplyResult
from app.agent.service import AgentService
from app.api.agent_schemas import (
    LogEvent,
    RejectRequest,
    RunCreate,
    RunDetail,
    RunList,
    RunLogs,
    RunSummary,
)
from app.api.deps import SettingsDep, require_api_key
from app.api.schemas import ErrorResponse
from app.api.tool_call import ERROR_STATUSES
from app.storage.runs import RunRecord

RUN_ID_PATTERN = r"^[0-9a-f]{16}$"
NOT_FOUND = "Run not found."

router = APIRouter(
    prefix="/api/v1/agent",
    tags=["agent"],
    dependencies=[Depends(require_api_key)],
    responses={code: {"model": ErrorResponse} for code in ERROR_STATUSES},
)

RunIdDep = Annotated[
    str,
    Path(pattern=RUN_ID_PATTERN, description="Run id: 16 lowercase hex characters."),
]


def get_agent_service(request: Request) -> AgentService:
    service: AgentService = request.app.state.agent_service
    return service


AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _to_summary(record: RunRecord) -> RunSummary:
    state = record.state
    patch_path = str(state.get("patch_path") or "")
    return RunSummary(
        id=record.id,
        task=record.task,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        test_status=str(state.get("test_status") or "not_run"),
        patch_path=patch_path,
        has_patch=bool(patch_path and state.get("patch_content")),
        error=record.error,
    )


def _to_detail(record: RunRecord) -> RunDetail:
    state = record.state
    summary = _to_summary(record)
    attempts = state.get("attempts")
    return RunDetail(
        **summary.model_dump(),
        analysis=str(state.get("analysis") or ""),
        plan=_strings(state.get("plan")),
        evidence=_strings(state.get("evidence")),
        inspected_files=_strings(state.get("inspected_files")),
        diff=str(state.get("diff") or ""),
        test_output=str(state.get("test_output") or ""),
        risks=_strings(state.get("risks")),
        review_ok=bool(state.get("review_ok")),
        attempts=attempts if isinstance(attempts, int) else 0,
        report=str(state.get("report") or ""),
        apply_result=(
            ApplyResult.model_validate(record.apply_result)
            if record.apply_result is not None
            else None
        ),
        reject_reason=record.reject_reason,
    )


def _load(service: AgentService, run_id: str) -> RunRecord:
    record = service.store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    return record


@router.post(
    "/runs",
    response_model=RunSummary,
    status_code=202,
    summary="Start an agent run",
    description=(
        "Queue a coding task and return immediately. The agent inspects the workspace, "
        "proposes at most one file change and runs the tests in the sandbox; it never writes "
        "to the workspace itself. Poll GET /api/v1/agent/runs/{run_id} until the status is "
        "awaiting_approval (then approve or reject) or completed."
    ),
    responses={429: {"model": ErrorResponse, "description": "Too many active runs."}},
)
async def create_run(
    body: RunCreate,
    background: BackgroundTasks,
    service: AgentServiceDep,
    settings: SettingsDep,
) -> RunSummary:
    if service.store.count_active() >= settings.max_active_runs:
        raise HTTPException(status_code=429, detail="Too many active runs. Try again shortly.")
    run_id = service.store.create_run(body.task)
    service.store.add_event(run_id, "info", "Run queued")
    record = _load(service, run_id)
    background.add_task(service.execute_run, run_id)
    return _to_summary(record)


@router.get(
    "/runs",
    response_model=RunList,
    summary="List agent runs",
    description="Most recent runs first. Summary fields only; fetch one run for the diff.",
)
async def list_runs(
    service: AgentServiceDep,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum runs to return.")] = 50,
) -> RunList:
    return RunList(runs=[_to_summary(record) for record in service.store.list_runs(limit)])


@router.get(
    "/runs/{run_id}",
    response_model=RunDetail,
    summary="Get one agent run",
    description=(
        "Full run view, including the diff and sandbox output, for the approval screen. "
        "Patch content and the workspace base hash are never returned."
    ),
    responses={404: {"model": ErrorResponse, "description": "Run not found."}},
)
async def get_run(service: AgentServiceDep, run_id: RunIdDep) -> RunDetail:
    return _to_detail(_load(service, run_id))


@router.get(
    "/runs/{run_id}/logs",
    response_model=RunLogs,
    summary="Read a run's log",
    description=(
        "Log events in insertion order. Poll with the returned next_after value as ?after= "
        "to receive only new lines; an empty page leaves next_after equal to the request."
    ),
    responses={404: {"model": ErrorResponse, "description": "Run not found."}},
)
async def get_run_logs(
    service: AgentServiceDep,
    run_id: RunIdDep,
    after: Annotated[int, Query(ge=0, description="Return events with a larger id.")] = 0,
    limit: Annotated[int, Query(ge=1, le=500, description="Maximum events to return.")] = 200,
) -> RunLogs:
    _load(service, run_id)
    events = service.store.list_events(run_id, after_id=after, limit=limit)
    return RunLogs(
        run_id=run_id,
        events=[LogEvent(**event) for event in events],
        next_after=int(events[-1]["id"]) if events else after,
    )


@router.post(
    "/runs/{run_id}/approve",
    response_model=RunDetail,
    summary="Approve and apply a run's patch",
    description=(
        "Write the patch stored with the run to the workspace. The request carries no patch "
        "data: the stored diff is bound to the file's hash, so it is refused if the file "
        "changed since the proposal. Only a run in awaiting_approval can be approved."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Run not found."},
        409: {"model": ErrorResponse, "description": "Run is not awaiting approval, or conflict."},
    },
)
async def approve_run(service: AgentServiceDep, run_id: RunIdDep) -> RunDetail:
    return _to_detail(await service.approve(run_id))


@router.post(
    "/runs/{run_id}/reject",
    response_model=RunDetail,
    summary="Reject a run's patch",
    description=(
        "Mark the patch rejected without touching the workspace. The body is optional and "
        "only records a short reason. Only a run in awaiting_approval can be rejected."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Run not found."},
        409: {"model": ErrorResponse, "description": "Run is not awaiting approval."},
    },
)
async def reject_run(
    service: AgentServiceDep,
    run_id: RunIdDep,
    body: RejectRequest | None = None,
) -> RunDetail:
    reason = body.reason if body is not None else ""
    return _to_detail(service.reject(run_id, reason))
