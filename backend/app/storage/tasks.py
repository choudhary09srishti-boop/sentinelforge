"""SQLite storage for agent tasks and their audit trail."""

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  task TEXT NOT NULL,
  status TEXT NOT NULL,
  state_json TEXT NOT NULL DEFAULT '{}',
  base_sha256 TEXT NOT NULL DEFAULT '',
  patch_id TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL REFERENCES tasks(id),
  event TEXT NOT NULL,
  detail TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_task ON audit_events(task_id);
"""

# queued -> running -> awaiting_approval | no_patch | failed
# awaiting_approval -> applying -> applied | failed, or -> rejected
STATUSES = (
    "queued",
    "running",
    "awaiting_approval",
    "no_patch",
    "failed",
    "applying",
    "applied",
    "rejected",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class TaskStore:
    """Tasks and audit events. Callers must pass already-sanitised text to add_event."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self._path, timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()

    def create_task(self, task: str) -> str:
        task_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tasks (id, task, status, created_at, updated_at) "
                "VALUES (?, ?, 'queued', ?, ?)",
                (task_id, task, now, now),
            )
        return task_id

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["state"] = json.loads(data.pop("state_json"))
        return data

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        state: dict[str, Any] | None = None,
        base_sha256: str | None = None,
        patch_id: str | None = None,
        error: str | None = None,
    ) -> None:
        if status is not None and status not in STATUSES:
            raise ValueError(f"Unknown task status: {status}")
        with self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET "
                "status = COALESCE(?, status), "
                "state_json = COALESCE(?, state_json), "
                "base_sha256 = COALESCE(?, base_sha256), "
                "patch_id = COALESCE(?, patch_id), "
                "error = COALESCE(?, error), "
                "updated_at = ? WHERE id = ?",
                (
                    status,
                    None if state is None else json.dumps(state),
                    base_sha256,
                    patch_id,
                    error,
                    _now(),
                    task_id,
                ),
            )

    def claim(self, task_id: str, expected: str, new: str) -> bool:
        """Atomically move a task from one status to another; False if it was not in `expected`.

        Approve and reject both claim awaiting_approval first, so a patch can only be
        applied once even if two requests arrive together.
        """
        if new not in STATUSES:
            raise ValueError(f"Unknown task status: {new}")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (new, _now(), task_id, expected),
            )
            return cur.rowcount == 1

    def add_event(self, task_id: str, event: str, detail: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_events (task_id, event, detail, created_at) VALUES (?, ?, ?, ?)",
                (task_id, event, detail, _now()),
            )

    def list_events(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event, detail, created_at FROM audit_events WHERE task_id = ? ORDER BY id",
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]
