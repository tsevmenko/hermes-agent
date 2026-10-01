"""Tests for gateway.kanban_owner_alerts (board-wide push of human-needed blocks)."""

from __future__ import annotations

import json
import time

import pytest

from gateway import kanban_owner_alerts as oa


def _cfg(**kw):
    raw = {"enabled": True, "platform": "telegram", "chat_id": "42", "boards": ["work"]}
    raw.update(kw)
    return oa.load_config(raw)


def test_config_defaults_to_home_channel(monkeypatch):
    monkeypatch.setattr(oa, "_HOME_CHANNELS", {})
    assert not oa.load_config({}).active  # no home channel known yet
    oa.set_home_channel("telegram", 777)
    c = oa.load_config({})
    assert c.active and c.chat_id == "777" and c.boards is None
    assert not oa.load_config({"enabled": False}).active
    assert oa.load_config({"chat_id": "5"}).chat_id == "5"  # explicit wins
    assert _cfg().active
    assert _cfg().block_kinds == {"needs_input", "capability"}


def test_respawn_guard_auto_blocks_are_silent():
    p = {"kind": "capability", "reason": "respawn guard 'active_pr' deferred this task 40 times"}
    assert oa.classify("blocked", p, _cfg()) is None
    assert oa.classify("block_loop_detected", p, _cfg()) is None
    assert oa.classify("blocked", p, _cfg(skip_system_guards=False)) == "capability"


@pytest.mark.parametrize(
    "event_kind,payload,expected",
    [
        ("blocked", {"kind": "needs_input"}, "needs_input"),
        ("blocked", {"kind": "capability"}, "capability"),
        ("blocked", {"kind": "transient"}, None),
        ("blocked", {"kind": "dependency"}, None),
        ("blocked", {"kind": None}, None),
        ("blocked", {}, None),
        ("block_loop_detected", {"kind": "needs_input"}, "triage"),
        ("status", {"status": "triage"}, "triage"),
        ("status", {"status": "blocked"}, None),
        ("respawn_guarded", {"reason": "active_pr"}, None),
        ("respawn_guard_exhausted", {"reason": "active_pr"}, None),
        ("unblocked", {}, None),
    ],
)
def test_classify(event_kind, payload, expected):
    assert oa.classify(event_kind, payload, _cfg()) == expected


def test_classify_triage_can_be_disabled():
    cfg = _cfg(triage=False)
    assert oa.classify("block_loop_detected", {}, cfg) is None
    assert oa.classify("status", {"status": "triage"}, cfg) is None


def test_format_keeps_question_and_options_drops_tried_and_ids():
    reason = (
        "uniSMM prod: remove QA rows? 1) yes, delete; 2) no, keep. "
        "Tried: 1) checked the thread; 2) read-only count."
    )
    alert = oa.Alert(
        board="work", event_id=7, task_id="t_deadbeef", event_kind="blocked",
        block_kind="needs_input", reason=reason, title="uniSMM *prod* cleanup",
        assignee="dev",
    )
    text = oa.format_alert(alert)
    lines = text.splitlines()
    assert lines[0] == "**uniSMM prod cleanup**"
    assert "@dev" in lines[1]
    assert "uniSMM prod: remove QA rows? 1) yes, delete; 2) no, keep." in text
    assert "Tried" not in text
    assert "t_deadbeef" not in text


def test_question_keeps_reason_without_tried():
    assert oa.question_from_reason("Pick: 1) A 2) B") == "Pick: 1) A 2) B"
    assert oa.question_from_reason("Tried: only this") == "Tried: only this"


def test_dedup_and_prune():
    state = {"cursors": {}, "sent": {oa.dedup_key("t1", "q"): time.time() - 3600 * 30}}
    oa.prune_sent(state, time.time(), 24)
    assert state["sent"] == {}


class _FakeKB:
    DEFAULT_BOARD = "default"

    def __init__(self, conn):
        self._conn = conn

    def list_boards(self, include_archived=False):
        return [{"slug": "work"}, {"slug": "other"}]

    def connect(self, board=None, strict_board_pin=True):
        return _NoClose(self._conn)


class _NoClose:
    def __init__(self, conn):
        self._c = conn

    def execute(self, *a, **k):
        return self._c.execute(*a, **k)

    def close(self):
        pass


@pytest.fixture
def board(monkeypatch):
    import sqlite3
    import hermes_cli.kanban_db as real

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE tasks (id TEXT, title TEXT, assignee TEXT, status TEXT)")
    conn.execute(
        "CREATE TABLE task_events (id INTEGER PRIMARY KEY, task_id TEXT, kind TEXT, payload TEXT)"
    )
    conn.execute("INSERT INTO tasks VALUES ('t1', 'Card one', 'dev', 'blocked')")
    fake = _FakeKB(conn)
    monkeypatch.setattr(real, "list_boards", fake.list_boards)
    monkeypatch.setattr(oa, "_connect", lambda kb, slug: fake.connect(board=slug))

    def add(kind, payload):
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload) VALUES ('t1', ?, ?)",
            (kind, json.dumps(payload)),
        )

    return conn, add


def test_collect_first_tick_starts_at_head_then_filters(board):
    conn, add = board
    add("blocked", {"kind": "needs_input", "reason": "old question"})
    cfg = _cfg()
    state = {"cursors": {}, "sent": {}}
    alerts, cursors = oa.collect(cfg, state)
    assert alerts == [] and cursors == {"work": 1}  # no history replay
    state["cursors"].update(cursors)

    add("blocked", {"kind": "transient", "reason": "flaky"})
    add("respawn_guarded", {"reason": "active_pr"})
    add("blocked", {"kind": "needs_input", "reason": "Q? 1) a 2) b"})
    add("unblocked", {})
    add("blocked", {"kind": "needs_input", "reason": "Q? 1) a 2) b"})  # same question again
    alerts, cursors = oa.collect(cfg, state)
    assert [a.event_id for a in alerts] == [4]
    assert alerts[0].block_kind == "needs_input"
    assert cursors == {"work": 6}

    # after a successful send the key is remembered → a re-block is silent
    state["cursors"].update(cursors)
    state["sent"][oa.dedup_key("t1", "Q? 1) a 2) b")] = time.time()
    add("blocked", {"kind": "needs_input", "reason": "Q? 1) a 2) b"})
    alerts, _ = oa.collect(cfg, state)
    assert alerts == []

    # a new question on the same card is a new alert
    add("blocked", {"kind": "capability", "reason": "need sudo"})
    state["cursors"]["work"] = 7
    alerts, _ = oa.collect(cfg, state)
    assert [a.block_kind for a in alerts] == ["capability"]


def test_state_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    oa.save_state({"cursors": {"work": 3}, "sent": {"k": 1.0}}, p)
    assert oa.load_state(p) == {"cursors": {"work": 3}, "sent": {"k": 1.0}}
    assert oa.load_state(tmp_path / "missing.json") == {"cursors": {}, "sent": {}}


def test_notifier_skips_event_covered_by_owner_alerts(monkeypatch):
    from types import SimpleNamespace
    from gateway import kanban_watchers_notifier as kw

    cfg = _cfg()
    monkeypatch.setattr(oa, "load_config", lambda raw=None: cfg)
    sub = {"platform": "telegram", "chat_id": "42", "thread_id": ""}
    ev = SimpleNamespace(payload={"kind": "needs_input"})
    assert kw._owner_alert_covers(sub, "blocked", ev, "work")
    assert not kw._owner_alert_covers(sub, "blocked", SimpleNamespace(payload={"kind": "transient"}), "work")
    assert not kw._owner_alert_covers(sub, "blocked", ev, "other")
    assert not kw._owner_alert_covers({**sub, "chat_id": "7"}, "blocked", ev, "work")
    assert not kw._owner_alert_covers(sub, "completed", ev, "work")
