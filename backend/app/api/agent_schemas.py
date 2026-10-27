"""Request and response models for the agent run API.

Patch content and the base hash are deliberately absent from every response model: a human
approves a diff, and the content used for the write comes from stored run state only.
"""

from pydantic import BaseModel, Field, field_validator

from app.agent.apply import ApplyResult

TASK_MIN_CHARS = 5
TASK_MAX_CHARS = 2000


class RunCreate(BaseModel):
    """A task to hand to the coding agent."""

    task: str = Field(
        min_length=TASK_MIN_CHARS,
        max_length=TASK_MAX_CHARS,
        description="What the agent should change, in one or two sentences.",
        examples=["Make add() actually add the two numbers in src/app.py"],
    )

    @field_validator("task")
    @classmethod
    def _strip_task(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < TASK_MIN_CHARS:
            raise ValueError(f"Task must be at least {TASK_MIN_CHARS} characters.")
        return stripped


class RunSummary(BaseModel):
    """List-row view of a run: no state blobs, no patch content."""

    id: str = Field(description="16 hex character run id.", examples=["9f2c1d4e5a6b7c8d"])
    task: str = Field(description="The task the agent was asked to do.")
    status: str = Field(
        description="queued, running, awaiting_approval, applied, rejected, completed or failed.",
        examples=["awaiting_approval"],
    )
    created_at: str = Field(description="ISO-8601 UTC creation time.")
    updated_at: str = Field(description="ISO-8601 UTC time of the last change.")
    test_status: str = Field(
        description=(
            "Sandbox result: not_run, passed, failed, no_tests, error, timeout, unavailable."
        ),
        examples=["passed"],
    )
    patch_path: str = Field(default="", description="File the run proposed to change, if any.")
    has_patch: bool = Field(description="True when the run holds a patch awaiting approval.")
    error: str = Field(default="", description="Failure reason, empty unless the run failed.")


class RunDetail(RunSummary):
    """Full run view for the approval screen."""

    analysis: str = Field(default="", description="The agent's one-sentence reading of the task.")
    plan: list[str] = Field(default_factory=list, description="Ordered plan steps.")
    evidence: list[str] = Field(default_factory=list, description="Citations gathered from docs.")
    inspected_files: list[str] = Field(default_factory=list, description="Workspace files read.")
    diff: str = Field(default="", description="Unified diff the human is approving.")
    test_output: str = Field(default="", description="Bounded sandbox output.")
    risks: list[str] = Field(default_factory=list, description="Risks the agent recorded.")
    review_ok: bool = Field(default=False, description="Whether the agent's self-review passed.")
    attempts: int = Field(default=0, description="How many patch attempts were made.")
    report: str = Field(default="", description="The agent's closing report.")
    apply_result: ApplyResult | None = Field(
        default=None, description="Set once a human approved and the patch was written."
    )
    reject_reason: str = Field(default="", description="Why a human rejected the patch.")


class RunList(BaseModel):
    """Newest runs first."""

    runs: list[RunSummary] = Field(default_factory=list)


class LogEvent(BaseModel):
    """One log line, in insertion order."""

    id: int = Field(description="Monotonic event id; use it as the paging cursor.")
    ts: str = Field(description="ISO-8601 UTC timestamp.")
    level: str = Field(description="info, warning or error.", examples=["info"])
    message: str = Field(description="Single-line summary; never file contents.")


class RunLogs(BaseModel):
    """A page of log events plus the cursor for the next poll."""

    run_id: str = Field(description="Run these events belong to.")
    events: list[LogEvent] = Field(default_factory=list)
    next_after: int = Field(
        description="Pass as ?after= to fetch only newer events. Equals the request when idle."
    )


class RejectRequest(BaseModel):
    """Optional note explaining a rejection."""

    reason: str = Field(
        default="",
        max_length=500,
        description="Short reason, shown to whoever reads the run later.",
        examples=["The patch also changes unrelated formatting."],
    )
