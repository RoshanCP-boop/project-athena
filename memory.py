"""
SQLite-backed long-term memory for Jarvis.

Three concepts:
  - facts: short pieces of information the user explicitly told Jarvis to
    remember ("my dog's name is Bella"). Always injected into the system
    prompt, so the LLM treats them as persistent context.
  - conversations: one row per session. Holds optional one-line summary
    written at the end of the session.
  - messages: every user/assistant message, scoped to a conversation.

Storage is local (~/sage/memory.db). Even after years of daily use it'll
sit comfortably under 50 MB.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
from contextlib import contextmanager
from datetime import datetime as dt
from typing import Iterator, Optional

DB_PATH = os.path.expanduser("~/sage/memory.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    content      TEXT    NOT NULL UNIQUE,
    created_at   TEXT    NOT NULL,
    last_used_at TEXT,
    hit_count    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT    NOT NULL,
    ended_at   TEXT,
    summary    TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
"""

# SQLite is thread-safe per-connection but not per-cursor. We open a fresh
# connection per call (cheap for SQLite) and serialise writes with a lock
# so we never hit "database is locked" under concurrent access from the
# audio thread + UI thread + processing thread.
_LOCK = threading.Lock()


def _now() -> str:
    return dt.now().isoformat(timespec="seconds")


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


def init_db() -> None:
    """Create the database file and schema if missing. Safe to call repeatedly."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _LOCK, _conn() as c:
        c.executescript(_SCHEMA)
        c.commit()


# ---------- facts -----------------------------------------------------------


def add_fact(content: str) -> bool:
    """Record a long-term fact. Returns True if newly added, False if duplicate."""
    content = content.strip()
    if not content:
        return False
    with _LOCK, _conn() as c:
        try:
            c.execute(
                "INSERT INTO facts (content, created_at) VALUES (?, ?)",
                (content, _now()),
            )
            c.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def remove_facts(query: str) -> int:
    """Delete facts matching `query` as case-insensitive substring.

    Returns number of rows deleted. If `query` is exactly a digit string,
    it's treated as an id instead.
    """
    query = query.strip()
    if not query:
        return 0
    with _LOCK, _conn() as c:
        if query.isdigit():
            cur = c.execute("DELETE FROM facts WHERE id = ?", (int(query),))
        else:
            cur = c.execute(
                "DELETE FROM facts WHERE LOWER(content) LIKE LOWER(?)",
                (f"%{query}%",),
            )
        c.commit()
        return cur.rowcount


def list_facts() -> list[str]:
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT content FROM facts ORDER BY id ASC"
        ).fetchall()
        return [r["content"] for r in rows]


def facts_for_prompt() -> str:
    """Return a string ready to drop into the system prompt, or '' if no facts."""
    facts = list_facts()
    if not facts:
        return ""
    body = "\n".join(f"- {f}" for f in facts)
    return (
        "Things you previously remembered about the user (treat as facts, "
        "don't restate verbatim unless asked):\n"
        f"{body}\n"
    )


# ---------- conversations & messages ----------------------------------------


def start_conversation() -> int:
    with _LOCK, _conn() as c:
        cur = c.execute(
            "INSERT INTO conversations (started_at) VALUES (?)", (_now(),)
        )
        c.commit()
        return cur.lastrowid


def end_conversation(conv_id: int, summary: Optional[str] = None) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "UPDATE conversations SET ended_at = ?, summary = ? WHERE id = ?",
            (_now(), summary, conv_id),
        )
        c.commit()


def add_message(conv_id: int, role: str, content: str) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conv_id, role, content, _now()),
        )
        c.commit()


def get_messages(conv_id: int) -> list[dict]:
    """Return all messages in a conversation, oldest first."""
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def recent_summaries(limit: int = 5) -> list[str]:
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT summary FROM conversations "
            "WHERE summary IS NOT NULL AND summary != '' "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r["summary"] for r in rows]


def summaries_for_prompt(limit: int = 3) -> str:
    sums = recent_summaries(limit)
    if not sums:
        return ""
    body = "\n".join(f"- {s}" for s in sums)
    return f"Earlier conversations (most recent first):\n{body}\n"


if __name__ == "__main__":
    # Tiny CLI for manual inspection: `python memory.py` lists facts.
    init_db()
    print("Facts:")
    for f in list_facts():
        print(f"  - {f}")
    print("\nRecent summaries:")
    for s in recent_summaries():
        print(f"  - {s}")
    print(f"\nDB: {DB_PATH}", file=sys.stderr)
