"""Connection + bootstrap helpers.

Primary store is Postgres 16 + pgvector (via docker compose). If it is not
reachable, the app degrades to a local SQLite file with the SAME logical
schema (embedding stored as JSON) so the project runs fully offline.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from waagent.config import settings

MIGRATIONS = Path(__file__).resolve().parent / "migrations"

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_at TEXT NOT NULL DEFAULT (datetime('now')),
    handoff INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    request_id TEXT,
    ts TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_messages_user_ts ON messages (user_id, ts);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL,
    source_msg_ids TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
    access_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_memories_user ON memories (user_id, archived);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    user_id TEXT,
    node TEXT NOT NULL,
    ms INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0,
    ts TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _pg_parts(url: str) -> dict:
    """postgresql+psycopg2://user:pass@host:port/db -> psycopg2 kwargs."""
    parsed = urlparse(url.replace("postgresql+psycopg2://", "postgresql://"))
    return {
        "dbname": parsed.path.lstrip("/") or "waagent",
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
    }


def pg_connect(timeout: int = 2):
    """Return a live psycopg2 connection or None (never raises)."""
    try:
        import psycopg2

        conn = psycopg2.connect(connect_timeout=timeout, **_pg_parts(settings.database_url))
        return conn
    except Exception:
        return None


def pg_available(timeout: int = 2) -> bool:
    conn = pg_connect(timeout)
    if conn is None:
        return False
    conn.close()
    return True


def init_postgres(conn) -> None:
    """Apply every migration file in order (idempotent - all IF NOT EXISTS)."""
    with conn.cursor() as cur:
        for path in sorted(MIGRATIONS.glob("*.sql")):
            cur.execute(path.read_text(encoding="utf-8"))
    conn.commit()


def sqlite_connect() -> sqlite3.Connection:
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SQLITE_SCHEMA)
    conn.commit()
    return conn


def ensure_ready() -> str:
    """Bootstrap whichever store is available. Returns 'postgres' | 'sqlite'."""
    conn = pg_connect()
    if conn is not None:
        init_postgres(conn)
        conn.close()
        return "postgres"
    sqlite_connect().close()
    return "sqlite"
