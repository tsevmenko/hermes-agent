"""``backlog``: a fully written card that is parked until someone takes it out.

Contract under test:

* the dispatcher never claims or spawns a backlog card;
* dependency promotion (``recompute_ready``) never moves it, even when its
  parents finish;
* the kernel's own recovery sweeps (stale claims, crashed workers, orphan
  reconciliation, unblock) leave it alone;
* ``backlog_task`` parks from todo/ready/blocked/triage/scheduled, and stops a
  running worker first; ``unbacklog_task`` returns it to ready (or todo while
  parents are open) and the next dispatcher tick spawns it;
* every move is audited with ``backlogged`` / ``unbacklogged`` events.
"""
from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

# Upstream splits connect/dispatch out of kanban_db; older trees keep them there.
try:  # pragma: no cover - layout shim
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
except ImportError:  # pragma: no cover
    kbc = kbd = kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch, all_assignees_spawnable):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kbc.init_db()
    return home


def _patch_pid_alive(monkeypatch, fn) -> None:
    # Liveness is read via the kanban_db facade upstream and via the local
    # module in older layouts; patch both so the test drives either.
    monkeypatch.setattr(kb, "_pid_alive", fn)
    monkeypatch.setattr(kbd, "_pid_alive", fn)


def _spawned_ids(res) -> list[str]:
    return [tid for (tid, _assignee, _ws) in res.spawned]


def _noop_spawn(*_a, **_k):
    return None


def _status(conn, tid: str) -> str:
    return kb.get_task(conn, tid).status


def _events(conn, tid: str, kinds=("backlogged", "unbacklogged")) -> list[tuple[str, dict]]:
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (tid,),
    ).fetchall()
    return [
        (r["kind"], json.loads(r["payload"]) if r["payload"] else {})
        for r in rows if r["kind"] in kinds
    ]


def test_backlog_is_a_valid_status_and_initial_status():
    assert "backlog" in kb.VALID_STATUSES
    assert "backlog" in kb.VALID_INITIAL_STATUSES


def test_create_in_backlog_is_never_dispatched(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="parked", assignee="dev", initial_status="backlog",
            created_by="pm",
        )
        assert _status(conn, tid) == "backlog"
        assert _events(conn, tid) == [("backlogged", {"from": "created", "author": "pm"})]
        assert kbd.has_spawnable_ready(conn) is False
        for _ in range(3):
            res = kbd.dispatch_once(conn, spawn_fn=_noop_spawn)
            assert tid not in _spawned_ids(res)
        assert kb.claim_task(conn, tid) is None
        assert _status(conn, tid) == "backlog"


def test_default_create_is_unchanged(kanban_home):
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="p", assignee="dev")
        a = kb.create_task(conn, title="a", assignee="dev")
        b = kb.create_task(conn, title="b", assignee="dev", parents=[parent])
        assert _status(conn, a) == "ready"
        assert _status(conn, b) == "todo"


def test_dependency_promotion_skips_backlog(kanban_home):
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="dev")
        child = kb.create_task(
            conn, title="child", assignee="dev", parents=[parent],
            initial_status="backlog",
        )
        assert _status(conn, child) == "backlog"
        assert kb.complete_task(conn, parent, result="ok")
        kb.recompute_ready(conn)
        assert _status(conn, child) == "backlog"
        res = kbd.dispatch_once(conn, spawn_fn=_noop_spawn)
        assert child not in _spawned_ids(res)
        assert _status(conn, child) == "backlog"


@pytest.mark.parametrize("source", ["todo", "ready", "blocked", "triage", "scheduled"])
def test_park_from_every_idle_status(kanban_home, source):
    with kbc.connect() as conn:
        if source == "triage":
            tid = kb.create_task(conn, title="t", assignee="dev", triage=True)
        elif source == "todo":
            parent = kb.create_task(conn, title="p", assignee="dev")
            tid = kb.create_task(conn, title="t", assignee="dev", parents=[parent])
        else:
            tid = kb.create_task(conn, title="t", assignee="dev")
            if source == "blocked":
                assert kb.block_task(conn, tid, reason="needs Anton")
            elif source == "scheduled":
                assert kb.schedule_task(conn, tid, reason="tomorrow")
        assert _status(conn, tid) == source
        assert kb.backlog_task(conn, tid, reason="later", author="pm")
        assert _status(conn, tid) == "backlog"
        kind, payload = _events(conn, tid)[-1]
        assert kind == "backlogged"
        assert payload["from"] == source
        assert payload["reason"] == "later"
        assert payload["author"] == "pm"
        # Parking twice is a no-op, not a second event.
        assert kb.backlog_task(conn, tid) is False
        assert len(_events(conn, tid)) == 1


@pytest.mark.parametrize("terminal", ["review", "done", "archived"])
def test_cannot_park_review_done_archived(kanban_home, terminal):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev")
        if terminal == "done":
            assert kb.complete_task(conn, tid, result="ok")
        elif terminal == "archived":
            assert kb.archive_task(conn, tid)
        else:
            assert kb.request_review(conn, tid, summary="s", force=True)
        assert _status(conn, tid) == terminal
        assert kb.backlog_task(conn, tid) is False
        assert _status(conn, tid) == terminal


def test_park_running_stops_worker_and_closes_run(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = kb.get_task(conn, tid).current_run_id
        lock = conn.execute("SELECT claim_lock FROM tasks WHERE id=?", (tid,)).fetchone()[0]
        conn.execute("UPDATE tasks SET worker_pid = 424242 WHERE id = ?", (tid,))
        conn.commit()
        sent: list[tuple[int, int]] = []
        alive = {"v": True}

        def fake_kill(pid, sig):
            sent.append((pid, sig))
            alive["v"] = False

        _patch_pid_alive(monkeypatch, lambda pid: alive["v"])
        assert kb.backlog_task(conn, tid, reason="park it", signal_fn=fake_kill)
        assert sent == [(424242, signal.SIGTERM)]
        t = kb.get_task(conn, tid)
        assert t.status == "backlog"
        assert t.current_run_id is None
        row = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?", (run_id,),
        ).fetchone()
        assert row["outcome"] == "backlogged" and row["ended_at"] is not None
        kind, payload = _events(conn, tid)[-1]
        assert kind == "backlogged" and payload["from"] == "running"
        assert payload["terminated"] is True and payload["prev_pid"] == 424242
        assert lock  # sanity: it really was claimed


def test_park_running_refused_when_worker_survives(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev")
        assert kb.claim_task(conn, tid) is not None
        conn.execute("UPDATE tasks SET worker_pid = 424243 WHERE id = ?", (tid,))
        conn.commit()
        _patch_pid_alive(monkeypatch, lambda pid: True)
        monkeypatch.setattr(kbd.time, "sleep", lambda s: None)
        assert kb.backlog_task(conn, tid, signal_fn=lambda pid, sig: None) is False
        assert _status(conn, tid) == "running"
        assert _events(conn, tid) == []


def test_kernel_sweeps_leave_backlog_alone(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev")
        assert kb.block_task(conn, tid, reason="x")
        assert kb.backlog_task(conn, tid)
        # A stale claim/pid left on the row must not make any sweep treat
        # it as a running card.
        conn.execute(
            "UPDATE tasks SET claim_expires = 1, worker_pid = 999999 WHERE id = ?",
            (tid,),
        )
        conn.commit()
        kb.release_stale_claims(conn)
        kbd.detect_crashed_workers(conn)
        kbd.reconcile_orphaned_running(conn)
        kb.recompute_ready(conn)
        assert kb.unblock_task(conn, tid) is False
        res = kbd.dispatch_once(conn, spawn_fn=_noop_spawn, stale_timeout_seconds=1)
        assert tid not in _spawned_ids(res)
        assert tid not in res.reconciled_orphans
        assert _status(conn, tid) == "backlog"


def test_round_trip_park_unpark_runs(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev")
        assert kb.backlog_task(conn, tid, author="pm")
        res = kbd.dispatch_once(conn, spawn_fn=_noop_spawn)
        assert tid not in _spawned_ids(res)
        assert kb.unbacklog_task(conn, tid, reason="go", author="pm")
        assert _status(conn, tid) == "ready"
        assert _events(conn, tid)[-1] == (
            "unbacklogged", {"status": "ready", "reason": "go", "author": "pm"},
        )
        res = kbd.dispatch_once(conn, spawn_fn=_noop_spawn)
        assert tid in _spawned_ids(res)
        assert _status(conn, tid) == "running"
        # Unbacklog on a non-backlog card is refused.
        assert kb.unbacklog_task(conn, tid) is False


def test_unbacklog_regates_on_open_parents(kanban_home):
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="p", assignee="dev")
        child = kb.create_task(
            conn, title="c", assignee="dev", parents=[parent], initial_status="backlog",
        )
        assert kb.unbacklog_task(conn, child)
        assert _status(conn, child) == "todo"
        assert kb.complete_task(conn, parent, result="ok")
        kb.recompute_ready(conn)
        assert _status(conn, child) == "ready"


def test_unbacklog_resets_failure_counter(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev")
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 5, last_failure_error = 'x' WHERE id = ?",
            (tid,),
        )
        conn.commit()
        assert kb.backlog_task(conn, tid)
        assert kb.unbacklog_task(conn, tid)
        row = conn.execute(
            "SELECT consecutive_failures, last_failure_error FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["consecutive_failures"] == 0 and row["last_failure_error"] is None


def test_list_filters_by_backlog(kanban_home):
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="a", assignee="dev", initial_status="backlog")
        kb.create_task(conn, title="b", assignee="dev")
        ids = [t.id for t in kb.list_tasks(conn, status="backlog")]
        assert ids == [a]


# --- CLI ------------------------------------------------------------------

def test_cli_backlog_unbacklog_and_list(kanban_home, capsys):
    from hermes_cli import kanban as kcli

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cli", assignee="dev")
    assert kcli.kanban_command(_ns(["backlog", tid, "--reason", "later"])) == 0
    out = capsys.readouterr().out
    assert f"{tid}: ready → backlog: later" in out
    assert kcli.kanban_command(_ns(["list", "--status", "backlog"])) == 0
    assert tid in capsys.readouterr().out
    assert kcli.kanban_command(_ns(["unbacklog", tid, "--json"])) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload == {"task_id": tid, "ok": True, "from": "backlog", "status": "ready"}
    # Unbacklog a ready card → exit 1.
    assert kcli.kanban_command(_ns(["unbacklog", tid])) == 1
    with kbc.connect() as conn:
        comments = [c.body for c in kb.list_comments(conn, tid)]
    assert "BACKLOG: later" in comments


def test_cli_create_initial_status_backlog(kanban_home, capsys):
    from hermes_cli import kanban as kcli

    assert kcli.kanban_command(
        _ns(["create", "parked via cli", "--assignee", "dev",
             "--initial-status", "backlog", "--json"])
    ) == 0
    created = json.loads(capsys.readouterr().out)
    with kbc.connect() as conn:
        assert _status(conn, created["id"]) == "backlog"


def _ns(argv):
    import argparse
    from hermes_cli import kanban as kcli

    try:  # upstream moved the argparse tree into kanban_parser
        from hermes_cli.kanban_parser import build_parser
    except ImportError:  # pragma: no cover
        build_parser = kcli.build_parser
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_parser(sub)
    return parser.parse_args(["kanban", *argv])


# --- tools ----------------------------------------------------------------

def test_tools_backlog_unbacklog_list_create(kanban_home, monkeypatch):
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "pm")
    r = json.loads(kt._handle_create({
        "title": "tool parked", "assignee": "dev", "initial_status": "backlog",
    }))
    assert r.get("status") == "backlog", r
    tid = r["task_id"]
    listed = json.loads(kt._handle_list({"status": "backlog"}))
    assert [t["id"] for t in listed["tasks"]] == [tid]
    r = json.loads(kt._handle_unbacklog({"task_id": tid, "reason": "go"}))
    assert r["status"] == "ready" and r["previous_status"] == "backlog", r
    r = json.loads(kt._handle_backlog({"task_id": tid}))
    assert r["status"] == "backlog" and r["previous_status"] == "ready", r
    r = json.loads(kt._handle_unbacklog({"task_id": "t_nope"}))
    assert "error" in r


def test_tools_refused_inside_worker(kanban_home, monkeypatch):
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    r = json.loads(kt._handle_backlog({"task_id": tid}))
    assert "error" in r
    with kbc.connect() as conn:
        assert _status(conn, tid) == "ready"


def test_tools_registered_in_kanban_toolset():
    import toolsets
    from tools import kanban_tools as kt  # noqa: F401 — registers tools
    from tools.registry import registry

    names = set(toolsets.TOOLSETS["kanban"]["tools"])
    assert {"kanban_backlog", "kanban_unbacklog"} <= names
    registered = set(registry.get_tool_names_for_toolset("kanban"))
    assert {"kanban_backlog", "kanban_unbacklog"} <= registered
    enum = kt.KANBAN_CREATE_SCHEMA["parameters"]["properties"]["initial_status"]["enum"]
    assert "backlog" in enum
    assert "backlog" in kt.KANBAN_LIST_SCHEMA["parameters"]["properties"]["status"]["enum"]
