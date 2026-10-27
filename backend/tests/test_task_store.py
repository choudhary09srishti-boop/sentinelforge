from pathlib import Path

import pytest

from app.storage.tasks import TaskStore


def make_store(tmp_path: Path) -> TaskStore:
    return TaskStore(tmp_path / "nested" / "tasks.db")


def test_create_and_get_task(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    task_id = store.create_task("fix the bug")
    row = store.get_task(task_id)
    assert row is not None
    assert row["task"] == "fix the bug"
    assert row["status"] == "queued"
    assert row["state"] == {}


def test_unknown_task_returns_none(tmp_path: Path) -> None:
    assert make_store(tmp_path).get_task("nope") is None


def test_update_task_stores_state_and_hashes(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    task_id = store.create_task("t")
    store.update_task(
        task_id,
        status="awaiting_approval",
        state={"diff": "x"},
        base_sha256="abc",
        patch_id="def",
    )
    row = store.get_task(task_id)
    assert row is not None
    assert row["status"] == "awaiting_approval"
    assert row["state"] == {"diff": "x"}
    assert row["base_sha256"] == "abc"
    assert row["patch_id"] == "def"


def test_update_rejects_unknown_status(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    task_id = store.create_task("t")
    with pytest.raises(ValueError):
        store.update_task(task_id, status="bogus")


def test_claim_succeeds_only_once(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    task_id = store.create_task("t")
    store.update_task(task_id, status="awaiting_approval")
    assert store.claim(task_id, "awaiting_approval", "applying") is True
    assert store.claim(task_id, "awaiting_approval", "applying") is False


def test_audit_events_keep_order(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    task_id = store.create_task("t")
    store.add_event(task_id, "created")
    store.add_event(task_id, "approved", "ok")
    events = store.list_events(task_id)
    assert [e["event"] for e in events] == ["created", "approved"]
    assert events[1]["detail"] == "ok"
