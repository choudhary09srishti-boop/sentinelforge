"""Demo helper: restore the sample workspace so the public demo can be run again."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import SettingsDep, require_api_key
from app.api.schemas import ErrorResponse
from app.api.tool_call import ERROR_STATUSES
from app.bootstrap import seed_workspace

router = APIRouter(
    prefix="/api/v1/demo",
    tags=["demo"],
    dependencies=[Depends(require_api_key)],
    responses={code: {"model": ErrorResponse} for code in ERROR_STATUSES},
)


class DemoResetResult(BaseModel):
    files_restored: int


@router.post("/reset", response_model=DemoResetResult)
def reset_demo_workspace(settings: SettingsDep) -> DemoResetResult:
    """Restore the sample workspace files (including the sample bug). Demo deployments only."""
    if not settings.bootstrap_samples:
        raise HTTPException(status_code=404, detail="Demo reset is not enabled.")
    return DemoResetResult(files_restored=seed_workspace(settings))
