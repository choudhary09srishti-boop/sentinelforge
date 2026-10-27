"""SQLite storage for agent runs and their log events.

One connection per operation keeps the store usable from several event loops (the API test
client may start a new one per request) and from worker threads. Status changes are single
conditional UPDATE statements, so a stale claim can never overwrite a newer status.
"""

import json
import re
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

RunStatus = Literal[
    "queued",
    "running",
    "awaiting_approval",
    "applied",
    "rejected",
    "completed",
    "failed",
]
LogLevel = Literal["info", "warning", "error"]

RUN_STATUSES: tuple[RunStatus, ...] = (
    "queued",
    "running",
    "awaiting_approval",
    "applied",
    "rejected",
    "completed",
    "failed",
)
ACTIVE_STATUSES: tuple[RunStatus, ...] = ("queued", "running")
LOG_LEVELS: tuple[LogLevel, ...] = ("info", "warning", "error")
UPDATABLE_FIELDS = frozenset({"state_json", "base_sha256", "error", "apply_json", "reject_reason"})
MAX_MESSAGE_CHARS = 500
INTERRUPTED_MESSAGE = "Server restarted before the run finished."
_RUN_ID = re.compile(r"^[0-9a-f]{16}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  task TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  state_json TEXT NOT NULL DEFAULT '{}',
  base_sha256 TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '',
  apply_json TEXT NOT NULL DEFAULT '',
  reject_reason TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS run_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  ts TEXT NOT NULL,
  level TEXT NOT NULL,
  message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, id);
"""

# Placeholders are static and unused status slots are bound to "" so the statement never
# needs string building and every value stays parameterised.
UPDATE_SQL = """
UPDATE runs SET
  status = :status,
  updated_at = :updated_at,
  state_json = COALESCE(:state_json, state_json),
  base_sha256 = COALESCE(:base_sha256, base_sha256),
  error = COALESCE(:error, error),
  apply_json = COALESCE(:apply_json, apply_json),
  reject_reason = COALESCE(:reject_reason, reject_reason)
WHERE id = :id AND status IN (:s0, :s1, :s2, :s3, :s4, :s5, :s6)
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _clean_message(message: str) -> str:
    """Collapse to a single line and bound the length so logs stay safe to display."""
    return " ".join(message.split())[:MAX_MESSAGE_CHARS]


class RunRecord(BaseModel):
    """One stored run. `state` is the agent state dict; `apply_result` the applied patch."""

    id: str
    task: str
    status: RunStatus
    created_at: str
    updated_at: str
    state: dict[str, Any] = Field(default_factory=dict)
    base_sha256: str = ""
    error: str = ""
    apply_result: dict[str, Any] | None = None
    reject_reason: str = ""


class RunStore:
    """Runs and their event log. Callers must pass already-sanitised text to add_event."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- validation ---------------------------------------------------------------------

    @staticmethod
    def _check_id(run_id: str) -> str:
        if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
            raise ValueError("Run id must be 16 lowercase hex characters.")
        return run_id

    @staticmethod
    def _check_status(status: str) -> RunStatus:
        if status not in RUN_STATUSES:
            raise ValueError(f"Unknown run status: {status}")
        return status

    @staticmethod
    def _check_level(level: str) -> LogLevel:
        if level not in LOG_LEVELS:
            raise ValueError(f"Unknown log level: {level}")
        return level

    # ---- runs ---------------------------------------------------------------------------

    def create_run(self, task: str) -> str:
        text = task.strip()
        if not text:
            raise ValueError("Task must not be empty.")
        run_id = secrets.token_hex(8)
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (id, task, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, text, "queued", now, now),
            )
        return run_id

    def get_run(self, run_id: str) -> RunRecord | None:
        self._check_id(run_id)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return None if row is None else _record(row)

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_record(row) for row in rows]

    def count_active(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM runs WHERE status IN (?, ?)", ACTIVE_STATUSES
            ).fetchone()
        return int(row["n"]) if row is not None else 0

    def transition(
        self, run_id: str, from_statuses: set[str], to_status: str, **fields: str
    ) -> bool:
        """Move a run atomically if it is still in one of `from_statuses`; False otherwise."""
        self._check_id(run_id)
        self._check_status(to_status)
        sources = [self._check_status(status) for status in sorted(from_statuses)]
        if not sources:
            raise ValueError("from_statuses must not be empty.")
        for name in fields:
            if name not in UPDATABLE_FIELDS:
                raise ValueError(f"Unknown run field: {name}")
        slots = [*sources, "", "", "", "", "", ""][:7]
        params: dict[str, Any] = {
            "id": run_id,
            "status": to_status,
            "updated_at": _now(),
            "state_json": None,
            "base_sha256": None,
            "error": None,
            "apply_json": None,
            "reject_reason": None,
        }
        params.update(fields)
        params.update({f"s{index}": value for index, value in enumerate(slots)})
        with self._connect() as conn:
            cursor = conn.execute(UPDATE_SQL, params)
            return cursor.rowcount == 1

    def recover_interrupted(self) -> int:
        """Fail runs a previous server process left queued or running."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM runs WHERE status IN (?, ?)", ACTIVE_STATUSES
            ).fetchall()
        recovered = 0
        for row in rows:
            run_id = str(row["id"])
            claimed = self.transition(
                run_id, set(ACTIVE_STATUSES), "failed", error=INTERRUPTED_MESSAGE
            )
            if claimed:
                self.add_event(run_id, "error", INTERRUPTED_MESSAGE)
                recovered += 1
        return recovered

    # ---- events -------------------------------------------------------------------------

    def add_event(self, run_id: str, level: str, message: str) -> None:
        self._check_id(run_id)
        safe_level = self._check_level(level)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO run_events (run_id, ts, level, message) VALUES (?, ?, ?, ?)",
                (run_id, _now(), safe_level, _clean_message(message)),
            )

    def list_events(self, run_id: str, after_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        self._check_id(run_id)
        if after_id < 0:
            raise ValueError("after_id must be >= 0.")
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500.")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, ts, level, message FROM run_events "
                "WHERE run_id = ? AND id > ? ORDER BY id LIMIT ?",
                (run_id, after_id, limit),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "ts": str(row["ts"]),
                "level": str(row["level"]),
                "message": str(row["message"]),
            }
            for row in rows
        ]


def _record(row: sqlite3.Row) -> RunRecord:
    raw_state = json.loads(str(row["state_json"]))
    raw_apply = str(row["apply_json"])
    return RunRecord(
        id=str(row["id"]),
        task=str(row["task"]),
        status=cast(RunStatus, str(row["status"])),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        state=raw_state if isinstance(raw_state, dict) else {},
        base_sha256=str(row["base_sha256"]),
        error=str(row["error"]),
        apply_result=json.loads(raw_apply) if raw_apply else None,
        reject_reason=str(row["reject_reason"]),
    )
