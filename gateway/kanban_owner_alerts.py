"""Board-wide owner alerts: push every human-needed kanban block to one chat.

The per-subscription notifier only reaches chats that subscribed to a given
task, and a subscription in ``wake`` mode never sends a visible message. A
card that blocks with ``kind='needs_input'`` therefore could sit unseen until
a pull-style digest picked it up. This module adds a push lane that is
independent of subscriptions:

* scans ``task_events`` of the configured boards with its own persistent
  cursor (one cursor per board, kept in ``<kanban_home>/owner_alerts_state.json``);
* alerts on ``blocked`` events whose ``kind`` is in ``block_kinds``
  (default ``needs_input`` / ``capability``), on ``block_loop_detected``
  (routed to triage) and on a ``status`` change into ``triage``;
* never alerts on transient / dependency / un-typed blocks, on
  ``respawn_guarded`` events, or on ``unblocked``;
* deduplicates by ``(task_id, reason)`` for ``dedup_hours`` so a respawn
  guard or a block → unblock → re-block cycle with the same question does not
  repeat the message;
* optionally mirrors the alert into the owner's chat session as a user-role
  turn, so the agent that owns that chat sees the same question and can act
  on the owner's answer.

Config (``kanban.owner_alerts`` in config.yaml; every key optional)::

    kanban:
      owner_alerts:
        enabled: true            # default true
        platform: telegram       # default telegram
        chat_id: "123456"        # default: the gateway's home channel
        boards: [work]           # default: every board
        block_kinds: [needs_input, capability]
        triage: true
        dedup_hours: 24
        mirror_to_session: true
        skip_system_guards: true # dispatcher "respawn guard …" auto-blocks stay silent

Inactive when no target chat is known (no ``chat_id`` and no home channel).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("gateway.run")

STATE_FILENAME = "owner_alerts_state.json"
_DEFAULT_BLOCK_KINDS = ("needs_input", "capability")
_MAX_REASON_CHARS = 1500
_MAX_TITLE_CHARS = 200
_TRIED_RE = re.compile(r"\s*\bTried\s*:", re.IGNORECASE)


@dataclass
class OwnerAlertConfig:
    enabled: bool = False
    platform: str = "telegram"
    chat_id: str = ""
    thread_id: str = ""
    boards: Optional[set[str]] = None
    block_kinds: set[str] = field(default_factory=lambda: set(_DEFAULT_BLOCK_KINDS))
    triage: bool = True
    dedup_hours: float = 24.0
    mirror_to_session: bool = True
    skip_system_guards: bool = True

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.chat_id and self.platform)


def load_config(raw: Any = None) -> OwnerAlertConfig:
    """Build the config from ``kanban.owner_alerts`` (or a given dict).

    Defaults: enabled, Telegram, the gateway's home channel for that platform
    (registered by the watcher via :func:`set_home_channel`), every board.
    """
    if raw is None:
        try:
            from hermes_cli.config import load_config as _load_cfg

            raw = ((_load_cfg() or {}).get("kanban") or {}).get("owner_alerts")
        except Exception:
            raw = None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        return OwnerAlertConfig()

    def _set(value: Any) -> Optional[set[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        items = {str(v).strip() for v in value if str(v).strip()}
        return items or None

    platform = str(raw.get("platform") or "telegram").strip().lower()
    home_chat, home_thread = _HOME_CHANNELS.get(platform, ("", ""))
    chat_id = str(raw.get("chat_id") or "").strip()
    thread_id = str(raw.get("thread_id") or "").strip()
    if not chat_id:
        chat_id, thread_id = home_chat, thread_id or home_thread
    cfg = OwnerAlertConfig(
        enabled=bool(raw.get("enabled", True)),
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        boards=_set(raw.get("boards")),
        triage=bool(raw.get("triage", True)),
        mirror_to_session=bool(raw.get("mirror_to_session", True)),
        skip_system_guards=bool(raw.get("skip_system_guards", True)),
    )
    kinds = _set(raw.get("block_kinds"))
    if kinds is not None:
        cfg.block_kinds = kinds
    try:
        cfg.dedup_hours = float(raw.get("dedup_hours", 24))
    except (TypeError, ValueError):
        cfg.dedup_hours = 24.0
    return cfg


_HOME_CHANNELS: dict[str, tuple[str, str]] = {}


def set_home_channel(platform: str, chat_id: Any, thread_id: Any = None) -> None:
    """Register the gateway's home channel (the default alert target)."""
    if chat_id:
        _HOME_CHANNELS[str(platform).lower()] = (str(chat_id), str(thread_id or ""))


# ---------------------------------------------------------------- state ----

def state_path() -> Path:
    from hermes_cli import kanban_db as _kb

    return _kb.kanban_home() / "kanban" / STATE_FILENAME


def load_state(path: Optional[Path] = None) -> dict:
    path = path or state_path()
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data.setdefault("cursors", {})
            data.setdefault("sent", {})
            return data
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("kanban owner alerts: unreadable state %s (%s); starting fresh", path, exc)
    return {"cursors": {}, "sent": {}}


def save_state(state: dict, path: Optional[Path] = None) -> None:
    path = path or state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".owner_alerts.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def prune_sent(state: dict, now: float, dedup_hours: float) -> None:
    horizon = now - max(dedup_hours, 0) * 3600
    sent = state.get("sent") or {}
    for key in [k for k, ts in sent.items() if not isinstance(ts, (int, float)) or ts < horizon]:
        sent.pop(key, None)


def dedup_key(task_id: str, reason: str) -> str:
    digest = hashlib.sha1((reason or "").strip().encode("utf-8")).hexdigest()[:16]
    return f"{task_id}:{digest}"


# ------------------------------------------------------------- filtering ----

@dataclass
class Alert:
    board: str
    event_id: int
    task_id: str
    event_kind: str
    block_kind: Optional[str]
    reason: str
    title: str
    assignee: str
    text: str = ""


_SYSTEM_GUARD_RE = re.compile(r"^\s*respawn guard\b", re.IGNORECASE)


def classify(event_kind: str, payload: Optional[dict], cfg: OwnerAlertConfig) -> Optional[str]:
    """Return the alert flavour for an event, or None to stay silent."""
    payload = payload or {}
    if cfg.skip_system_guards and _SYSTEM_GUARD_RE.match(str(payload.get("reason") or "")):
        # Dispatcher respawn-guard auto-blocks (e.g. 'active_pr') are not an
        # agent's question; they flood in bursts and stay on the digest lane.
        return None
    if event_kind == "blocked":
        kind = payload.get("kind")
        if kind and str(kind) in cfg.block_kinds:
            return str(kind)
        return None
    if event_kind == "block_loop_detected":
        return "triage" if cfg.triage else None
    if event_kind == "status":
        if cfg.triage and str(payload.get("status") or "") == "triage":
            return "triage"
        return None
    return None


def question_from_reason(reason: str) -> str:
    """The agent's question verbatim, without the trailing ``Tried:`` log."""
    text = (reason or "").strip()
    m = _TRIED_RE.search(text)
    if m and m.start() > 0:
        text = text[: m.start()].rstrip()
    if len(text) > _MAX_REASON_CHARS:
        text = text[: _MAX_REASON_CHARS - 1].rstrip() + "…"
    return text


_NEED_LINE = {
    "needs_input": "Нужно твоё решение — агент{who} остановился и ждёт ответа.",
    "capability": "Нужен доступ или действие, которое агент{who} сам сделать не может.",
    "triage": "Карточка ушла в triage после повторных блоков — нужно твоё решение.",
}


def _md_safe(text: str) -> str:
    # The Telegram adapter converts standard markdown to MarkdownV2; keep the
    # title's own asterisks/underscores from breaking the bold wrapper.
    return text.replace("*", "").replace("`", "'")


def format_alert(alert: Alert) -> str:
    title = _md_safe((alert.title or "Без названия").strip())[:_MAX_TITLE_CHARS]
    flavour = alert.block_kind if alert.block_kind in _NEED_LINE else "needs_input"
    who = f" @{alert.assignee}" if alert.assignee and flavour != "triage" else ""
    lines = [f"**{title}**", _NEED_LINE[flavour].format(who=who)]
    question = question_from_reason(alert.reason)
    if question:
        lines += ["", question]
    lines += ["", "Ответь здесь — PM передаст ответ агенту и разблокирует карточку."]
    return "\n".join(lines)


def _rows(conn, cursor: int, limit: int = 500) -> Iterable[Any]:
    return conn.execute(
        "SELECT e.id, e.task_id, e.kind, e.payload, t.title, t.assignee, t.status "
        "FROM task_events e LEFT JOIN tasks t ON t.id = e.task_id "
        "WHERE e.id > ? AND e.kind IN ('blocked', 'block_loop_detected', 'status') "
        "ORDER BY e.id LIMIT ?",
        (int(cursor), int(limit)),
    ).fetchall()


def _connect(kb: Any, slug: str):
    """Open a board DB across kanban_db layouts (monolith or split modules)."""
    connect = getattr(kb, "connect", None)
    if connect is None:
        from hermes_cli import kanban_db_connect as _kbc

        connect = _kbc.connect
    try:
        return connect(board=slug, strict_board_pin=False)
    except TypeError:
        return connect(board=slug)


def collect(cfg: OwnerAlertConfig, state: dict, now: Optional[float] = None) -> tuple[list[Alert], dict]:
    """Scan boards for new alert-worthy events.

    Returns ``(alerts, new_cursors)``. Cursors are NOT written into ``state``
    here: the caller commits them only after delivery succeeds, so a failed
    send is retried on the next tick. Events that are not alert-worthy (or
    deduplicated) are still covered by the returned cursor.
    """
    from hermes_cli import kanban_db as _kb

    now = time.time() if now is None else now
    alerts: list[Alert] = []
    new_cursors: dict[str, int] = {}
    cursors = state.setdefault("cursors", {})
    sent = state.setdefault("sent", {})
    try:
        boards = _kb.list_boards(include_archived=False)
    except Exception:
        boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
    seen_keys: set[str] = set()
    for meta in boards:
        slug = meta.get("slug") or _kb.DEFAULT_BOARD
        if cfg.boards is not None and slug not in cfg.boards:
            continue
        try:
            conn = _connect(_kb, slug)
        except Exception as exc:
            logger.debug("kanban owner alerts: cannot open board %s: %s", slug, exc)
            continue
        try:
            if slug not in cursors:
                # First sight of this board: start from "now" instead of
                # replaying the whole history into the owner's chat.
                row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM task_events").fetchone()
                new_cursors[slug] = int(row[0] or 0)
                logger.info(
                    "kanban owner alerts: initialised cursor for board %s at event %s",
                    slug, new_cursors[slug],
                )
                continue
            cursor = int(cursors.get(slug) or 0)
            last = cursor
            for row in _rows(conn, cursor):
                ev_id, task_id, ev_kind, payload_raw, title, assignee, _status = row
                last = max(last, int(ev_id))
                try:
                    payload = json.loads(payload_raw) if payload_raw else {}
                except Exception:
                    payload = {}
                flavour = classify(ev_kind, payload, cfg)
                if not flavour:
                    continue
                reason = str(payload.get("reason") or "")
                key = dedup_key(task_id, reason)
                if key in sent or key in seen_keys:
                    logger.info(
                        "kanban owner alerts: skip duplicate %s event %s for %s (same question already sent)",
                        ev_kind, ev_id, task_id,
                    )
                    continue
                seen_keys.add(key)
                alert = Alert(
                    board=slug,
                    event_id=int(ev_id),
                    task_id=str(task_id),
                    event_kind=str(ev_kind),
                    block_kind=flavour,
                    reason=reason,
                    title=str(title or ""),
                    assignee=str(assignee or ""),
                )
                alert.text = format_alert(alert)
                alerts.append(alert)
            new_cursors[slug] = last
        except Exception as exc:
            logger.warning("kanban owner alerts: scan failed on board %s: %s", slug, exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass
    return alerts, new_cursors


def mirror_alert(cfg: OwnerAlertConfig, alert: Alert) -> bool:
    """Put the alert into the owner's chat session as a user-role turn."""
    try:
        from gateway.mirror import mirror_to_session

        text = (
            f"[Kanban owner alert: {alert.task_id} on board {alert.board}, "
            f"{alert.event_kind}/{alert.block_kind}, assignee @{alert.assignee or '-'}]\n"
            f"{alert.text}\n\n"
            f"Full block reason: {alert.reason}\n"
            "This question was already pushed to Anton in this chat. When he answers it, "
            f"hand the answer to the card (kanban_comment + kanban_unblock on {alert.task_id}) "
            "and confirm the card moved; do not repeat the question."
        )
        return bool(mirror_to_session(
            cfg.platform,
            cfg.chat_id,
            text,
            source_label="kanban",
            thread_id=cfg.thread_id or None,
            role="user",
        ))
    except Exception as exc:
        logger.warning("kanban owner alerts: mirror failed for %s: %s", alert.task_id, exc)
        return False
