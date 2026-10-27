from pathlib import Path

import pytest

from app.storage.runs import RunStore


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "data" / "runs.sqlite3")


def test_create_and_get_run(store: RunStore) -> None:
    run_id = store.create_run("  Fix the calculator  ")
    assert len(run_id) == 16
    record = store.get_run(run_id)
    assert record is not None
    assert record.task == "Fix the calculator"
    assert record.status == "queued"
    assert record.state == {}
    assert record.apply_result is None
    assert record.base_sha256 == ""
    assert record.error == ""
    assert record.created_at and record.updated_at


def test_get_unknown_run_returns_none(store: RunStore) -> None:
    assert store.get_run("0" * 16) is None


def test_list_runs_is_newest_first(store: RunStore) -> None:
    first = store.create_run("First task for the agent")
    second = store.create_run("Second task for the agent")
    third = store.create_run("Third task for the agent")
    ids = [record.id for record in store.list_runs()]
    assert ids == [third, second, first]


def test_list_runs_respects_limit(store: RunStore) -> None:
    ids = [store.create_run(f"Task number {n} for the agent") for n in range(4)]
    limited = [record.id for record in store.list_runs(limit=2)]
    assert limited == [ids[3], ids[2]]


def test_create_run_rejects_blank_task(store: RunStore) -> None:
    with pytest.raises(ValueError):
        store.create_run("   ")


def test_events_are_ordered_and_paged(store: RunStore) -> None:
    run_id = store.create_run("Task with several log lines")
    for number in range(5):
        store.add_event(run_id, "info", f"step {number}")
    events = store.list_events(run_id)
    assert [event["message"] for event in events] == [f"step {n}" for n in range(5)]
    assert [event["id"] for event in events] == sorted(event["id"] for event in events)

    first_page = store.list_events(run_id, limit=2)
    assert [event["message"] for event in first_page] == ["step 0", "step 1"]
    second_page = store.list_events(run_id, after_id=first_page[-1]["id"], limit=2)
    assert [event["message"] for event in second_page] == ["step 2", "step 3"]


def test_events_after_last_id_are_empty(store: RunStore) -> None:
    run_id = store.create_run("Task whose log is drained")
    store.add_event(run_id, "info", "only line")
    last = store.list_events(run_id)[-1]["id"]
    assert store.list_events(run_id, after_id=last) == []


def test_add_event_truncates_and_flattens_message(store: RunStore) -> None:
    run_id = store.create_run("Task with a huge log line")
    store.add_event(run_id, "error", "x" * 900 + "\nsecond line")
    event = store.list_events(run_id)[0]
    assert len(event["message"]) == 500
    assert "\n" not in event["message"]


def test_add_event_rejects_unknown_level(store: RunStore) -> None:
    run_id = store.create_run("Task with a bad log level")
    with pytest.raises(ValueError):
        store.add_event(run_id, "debug", "nope")


def test_transition_is_atomic(store: RunStore) -> None:
    run_id = store.create_run("Task that moves through states")
    assert store.transition(run_id, {"queued"}, "running") is True
    assert store.transition(run_id, {"queued"}, "running") is False
    assert store.transition(run_id, {"awaiting_approval"}, "applied") is False
    record = store.get_run(run_id)
    assert record is not None
    assert record.status == "running"


def test_transition_updates_only_allowed_fields(store: RunStore) -> None:
    run_id = store.create_run("Task that stores a patch hash")
    moved = store.transition(
        run_id, {"queued"}, "awaiting_approval", base_sha256="abc", state_json='{"a": 1}'
    )
    assert moved is True
    record = store.get_run(run_id)
    assert record is not None
    assert record.base_sha256 == "abc"
    assert record.state == {"a": 1}


def test_transition_rejects_unknown_field(store: RunStore) -> None:
    run_id = store.create_run("Task with an illegal update field")
    with pytest.raises(ValueError):
        store.transition(run_id, {"queued"}, "running", task="hijacked")


def test_transition_rejects_unknown_status(store: RunStore) -> None:
    run_id = store.create_run("Task with an illegal target status")
    with pytest.raises(ValueError):
        store.transition(run_id, {"queued"}, "exploded")


def test_transition_rejects_malformed_id(store: RunStore) -> None:
    with pytest.raises(ValueError):
        store.transition("not-a-run-id", {"queued"}, "running")


def test_count_active_counts_queued_and_running(store: RunStore) -> None:
    queued = store.create_run("Queued task")
    running = store.create_run("Running task")
    store.transition(running, {"queued"}, "running")
    finished = store.create_run("Finished task")
    store.transition(finished, {"queued"}, "completed")
    assert store.count_active() == 2
    store.transition(queued, {"queued"}, "failed", error="boom")
    assert store.count_active() == 1


def test_recover_interrupted_only_touches_active_runs(store: RunStore) -> None:
    queued = store.create_run("Queued when the server died")
    running = store.create_run("Running when the server died")
    store.transition(running, {"queued"}, "running")
    waiting = store.create_run("Waiting for a human")
    store.transition(waiting, {"queued"}, "awaiting_approval")
    done = store.create_run("Already done")
    store.transition(done, {"queued"}, "completed")

    assert store.recover_interrupted() == 2

    for run_id in (queued, running):
        record = store.get_run(run_id)
        assert record is not None
        assert record.status == "failed"
        assert record.error == "Server restarted before the run finished."
        events = store.list_events(run_id)
        assert [event["level"] for event in events] == ["error"]
    for run_id in (waiting, done):
        record = store.get_run(run_id)
        assert record is not None
        assert record.status != "failed"


def test_data_survives_reopening_the_same_file(tmp_path: Path) -> None:
    path = tmp_path / "data" / "runs.sqlite3"
    first = RunStore(path)
    run_id = first.create_run("Task that must survive a restart")
    first.add_event(run_id, "info", "Run queued")
    first.transition(run_id, {"queued"}, "completed", state_json='{"test_status": "passed"}')

    second = RunStore(path)
    record = second.get_run(run_id)
    assert record is not None
    assert record.status == "completed"
    assert record.state == {"test_status": "passed"}
    assert [event["message"] for event in second.list_events(run_id)] == ["Run queued"]
