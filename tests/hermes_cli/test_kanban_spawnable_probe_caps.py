"""The "dispatcher stuck" health probe must honour the dispatcher's caps.

``has_spawnable_ready`` feeds the gateway's stuck warning ("ready queue
non-empty ... but 0 workers spawned"). Ready work the tick deliberately holds
back — its assignee is at ``kanban.max_in_progress_per_profile``, or the host is
at ``kanban.max_in_progress`` — is correctly idle, so the probe must not report
it as spawnable. Each case is checked against a real ``dispatch_once`` dry run
on the same board so probe and dispatcher cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch, all_assignees_spawnable):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _mark_running(conn, task_id: str) -> None:
    # A real claim, so orphan reconciliation in dispatch_once keeps it running.
    assert kb.claim_task(conn, task_id) is not None


def _ready_ids(conn) -> list[str]:
    return [r["id"] for r in conn.execute("SELECT id FROM tasks WHERE status = 'ready'")]


def _board(conn, running: dict[str, int], ready: dict[str, int]) -> None:
    for assignee, n in running.items():
        for i in range(n):
            _mark_running(conn, kb.create_task(conn, title=f"run {assignee} {i}", assignee=assignee))
    for assignee, n in ready.items():
        for i in range(n):
            kb.create_task(conn, title=f"ready {assignee} {i}", assignee=assignee)
    kb.recompute_ready(conn)


def test_per_profile_capped_ready_is_not_spawnable(kanban_home):
    """3 ready delivery-qa cards while delivery-qa already runs 2 (cap 2)."""
    with kbc.connect() as conn:
        _board(conn, running={"delivery-qa": 2}, ready={"delivery-qa": 3})
        assert len(_ready_ids(conn)) == 3
        # Legacy call (no caps) still sees spawnable work.
        assert kbd.has_spawnable_ready(conn) is True
        assert kbd.has_spawnable_ready(conn, max_in_progress_per_profile=2) is False
        res = kbd.dispatch_once(conn, dry_run=True, max_in_progress_per_profile=2)
    assert res.spawned == []
    assert len(res.skipped_per_profile_capped) == 3


def test_one_uncapped_assignee_keeps_probe_true(kanban_home):
    """A capped profile must not mask a ready card whose profile has room."""
    with kbc.connect() as conn:
        _board(conn, running={"delivery-qa": 2}, ready={"delivery-qa": 2, "infra": 1})
        assert kbd.has_spawnable_ready(conn, max_in_progress_per_profile=2) is True
        res = kbd.dispatch_once(conn, dry_run=True, max_in_progress_per_profile=2)
    assert [a for _t, a, _w in res.spawned] == ["infra"]


def test_global_cap_reached_is_not_spawnable(kanban_home):
    """Host at kanban.max_in_progress: dispatcher spawns nothing, probe is idle."""
    with kbc.connect() as conn:
        _board(conn, running={"dev": 1, "infra": 1}, ready={"sage": 1})
        assert kbd.has_spawnable_ready(conn, max_in_progress=2) is False
        assert kbd.has_spawnable_ready(conn, max_in_progress=3) is True
        res = kbd.dispatch_once(conn, dry_run=True, max_in_progress=2)
    assert res.spawned == []


def test_review_probe_honours_per_profile_cap(kanban_home):
    with kbc.connect() as conn:
        _board(conn, running={"delivery-qa": 2}, ready={})
        tid = kb.create_task(conn, title="to review", assignee="delivery-qa")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        assert kbd.has_spawnable_review(conn) is True
        assert kbd.has_spawnable_review(conn, max_in_progress_per_profile=2) is False
        assert kbd.has_spawnable_review(conn, max_in_progress_per_profile=3) is True


@pytest.mark.parametrize("bad", [None, 0, -1, True, "2"])
def test_unusable_caps_are_ignored(kanban_home, bad):
    """Invalid caps behave like the dispatcher: not applied."""
    with kbc.connect() as conn:
        _board(conn, running={"delivery-qa": 2}, ready={"delivery-qa": 1})
        assert kbd.has_spawnable_ready(
            conn, max_in_progress=bad, max_in_progress_per_profile=bad,
        ) is True


def test_gateway_probe_passes_dispatcher_caps(kanban_home):
    """The gateway's ready_nonempty uses the settings the tick runs with."""
    from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher

    with kbc.connect() as conn:
        _board(conn, running={"delivery-qa": 2}, ready={"delivery-qa": 3})

    def settings(per_profile):
        return _DispatcherSettings(
            interval=60.0, max_spawn=None, max_in_progress=12, failure_limit=2,
            stale_timeout_seconds=0, reconcile_orphans=True, default_assignee=None,
            max_in_progress_per_profile=per_profile,
        )

    assert _KanbanDispatcher(kb, settings(2)).ready_nonempty() is False
    assert _KanbanDispatcher(kb, settings(None)).ready_nonempty() is True
