"""Memory pipeline: short-term (session) + long-term (semantic).

Short-term : Redis key per user -> turn buffer + rolling summary + flags.
Long-term  : Postgres+pgvector (or SQLite fallback) -> durable user facts,
             retrieved by cosine similarity and injected into the prompt.

The write policy (classify_memory / should_remember) is a PURE function, so
persistence decisions are unit-testable and the LLM never decides on its own.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from typing import Callable, Optional

from waagent.config import settings
from waagent.schemas import Memory, MemoryKind, utcnow

# --------------------------------------------------------------------------- #
# PII stripping + write policy
# --------------------------------------------------------------------------- #
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s]{7,}\d)(?!\d)")

_PREFERENCE = ("i prefer", "i like", "i always", "call me", "my language", "speak", "don't call", "do not call", "i hate")
_EVENT = ("my order", "i ordered", "my ticket", "my complaint", "my refund", "my payment", "i paid", "my account", "last week", "yesterday")
_SENTIMENT = ("frustrat", "angry", "upset", "annoyed", "love this", "very happy", "disappointed")
_FACT = ("my name is", "i am ", "i'm ", "i live", "my email", "i work", "my number", "i have a")
_CHITCHAT = ("thanks", "thank you", "hello", "hey", "okay", "bye", "good morning", "good evening")
_QUESTION_STARTS = ("where", "what", "when", "who", "why", "how", "which",
                    "do", "does", "can", "is", "are", "should", "will", "could", "would")


def strip_pii(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _CARD.sub("[card]", text)
    text = _PHONE.sub("[phone]", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_memory(text: str) -> Optional[MemoryKind]:
    """Return the kind if the message is worth long-term memory, else None."""
    t = strip_pii(text).lower().strip()
    if len(t) < 12:
        return None
    if t.endswith("?"):
        return None  # a question is not a fact about the user
    if t.split()[0] in _QUESTION_STARTS:
        return None  # interrogatives are not durable facts
    words = t.split()
    if len(words) <= 4 and any(g in t for g in _CHITCHAT):
        return None
    for hint in _PREFERENCE:
        if hint in t:
            return MemoryKind.preference
    for hint in _EVENT:
        if hint in t:
            return MemoryKind.event
    for hint in _SENTIMENT:
        if hint in t:
            return MemoryKind.sentiment
    for hint in _FACT:
        if hint in t:
            return MemoryKind.fact
    return None


def should_remember(text: str) -> bool:
    return classify_memory(text) is not None


# --------------------------------------------------------------------------- #
# Short-term: session store (Redis, dict fallback)
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return utcnow().isoformat()


def _empty_state() -> dict:
    return {"turns": [], "summary": "", "flags": {}}


class SessionStore:
    def __init__(self, ttl_seconds: int = 86400) -> None:
        self._ttl = ttl_seconds
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._redis = None
        try:
            import redis

            client = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            client.ping()
            self._redis = client
        except Exception:
            self._redis = None

    @property
    def backend(self) -> str:
        return "redis" if self._redis else "dict"

    def _key(self, user_id: str) -> str:
        return f"session:{user_id}"

    def _load(self, user_id: str) -> dict:
        if self._redis:
            raw = self._redis.get(self._key(user_id))
            return json.loads(raw) if raw else _empty_state()
        with self._lock:
            return json.loads(json.dumps(self._mem.get(user_id, _empty_state())))

    def _save(self, user_id: str, state: dict) -> None:
        if self._redis:
            self._redis.set(self._key(user_id), json.dumps(state), ex=self._ttl)
            return
        with self._lock:
            self._mem[user_id] = state

    def get(self, user_id: str) -> dict:
        return self._load(user_id)

    def get_turns(self, user_id: str) -> list[dict]:
        return self._load(user_id).get("turns", [])

    def get_summary(self, user_id: str) -> str:
        return self._load(user_id).get("summary", "")

    def append_turn(self, user_id: str, role: str, text: str) -> None:
        state = self._load(user_id)
        state.setdefault("turns", []).append({"role": role, "text": text, "ts": _now_iso()})
        self._save(user_id, state)

    def set_summary(self, user_id: str, summary: str) -> None:
        state = self._load(user_id)
        state["summary"] = summary
        self._save(user_id, state)

    def get_flags(self, user_id: str) -> dict:
        return self._load(user_id).get("flags", {})

    def set_flag(self, user_id: str, key: str, value) -> None:
        state = self._load(user_id)
        state.setdefault("flags", {})[key] = value
        self._save(user_id, state)

    def clear(self, user_id: str) -> None:
        if self._redis:
            self._redis.delete(self._key(user_id))
            return
        with self._lock:
            self._mem.pop(user_id, None)


# --------------------------------------------------------------------------- #
# Rolling-summary compaction
# --------------------------------------------------------------------------- #
def default_summarizer(prior_summary: str, old_turns: list[dict]) -> str:
    body = "; ".join(f"{t.get('role', 'user')}: {t.get('text', '')}" for t in old_turns)
    combined = f"{prior_summary} {body}".strip() if prior_summary else body
    return combined[:800]


def needs_compaction(session: SessionStore, user_id: str) -> bool:
    return len(session.get_turns(user_id)) > settings.summary_trigger_turns


def compact(
    session: SessionStore,
    user_id: str,
    summarize_fn: Callable[[str, list[dict]], str] = default_summarizer,
) -> str:
    """Fold the oldest turns into the rolling summary, keep recent turns raw."""
    state = session.get(user_id)
    turns = state.get("turns", [])
    if len(turns) <= settings.summary_trigger_turns:
        return state.get("summary", "")
    keep = max(2, settings.session_max_turns // 2)
    old, recent = turns[:-keep], turns[-keep:]
    state["summary"] = summarize_fn(state.get("summary", ""), old)
    state["turns"] = recent
    session._save(user_id, state)
    return state["summary"]


# --------------------------------------------------------------------------- #
# Long-term: semantic stores
# --------------------------------------------------------------------------- #
def _vec_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"


class PostgresSemanticStore:
    backend = "postgres"

    def __init__(self) -> None:
        from db.database import pg_connect

        self._connect = pg_connect

    def add(self, mem: Memory, embedding: list[float]) -> None:
        conn = self._connect()
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories (id, user_id, kind, content, embedding,
                                      created_at, last_seen_at, access_count)
                VALUES (%s,%s,%s,%s,%s::vector,%s,%s,%s)
                ON CONFLICT (id) DO UPDATE
                    SET content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        last_seen_at = now()
                """,
                (
                    mem.id, mem.user_id, mem.kind.value, mem.content,
                    _vec_literal(embedding), mem.created_at, mem.last_seen_at, mem.access_count,
                ),
            )
        conn.close()

    def search(self, user_id: str, query_embedding: list[float], k: int) -> list[Memory]:
        conn = self._connect()
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, kind, content, created_at, last_seen_at, access_count,
                       1 - (embedding <=> %s::vector) AS score
                FROM memories
                WHERE user_id = %s AND archived = false
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (_vec_literal(query_embedding), user_id, _vec_literal(query_embedding), k),
            )
            rows = cur.fetchall()
        conn.close()
        return [
            Memory(id=r[0], user_id=r[1], kind=r[2], content=r[3],
                   created_at=r[4], last_seen_at=r[5], access_count=r[6], score=float(r[7]))
            for r in rows
        ]

    def touch(self, ids: list[str]) -> None:
        if not ids:
            return
        conn = self._connect()
        with conn, conn.cursor() as cur:
            cur.execute("UPDATE memories SET access_count = access_count + 1 WHERE id = ANY(%s)", (ids,))
        conn.close()


class SqliteSemanticStore:
    backend = "sqlite"

    def __init__(self) -> None:
        from db.database import sqlite_connect

        self._connect = sqlite_connect

    def add(self, mem: Memory, embedding: list[float]) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO memories (id, user_id, kind, content, embedding, created_at, last_seen_at, access_count)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                content = excluded.content, embedding = excluded.embedding, last_seen_at = excluded.last_seen_at
            """,
            (mem.id, mem.user_id, mem.kind.value, mem.content, json.dumps(embedding),
             mem.created_at.isoformat(), mem.last_seen_at.isoformat(), mem.access_count),
        )
        conn.commit()
        conn.close()

    def search(self, user_id: str, query_embedding: list[float], k: int) -> list[Memory]:
        from waagent.embeddings import cosine

        conn = self._connect()
        rows = conn.execute(
            "SELECT id,user_id,kind,content,created_at,last_seen_at,access_count,embedding "
            "FROM memories WHERE user_id=? AND archived=0",
            (user_id,),
        ).fetchall()
        conn.close()
        scored = []
        for r in rows:
            score = cosine(query_embedding, json.loads(r["embedding"]))
            scored.append((score, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            Memory(id=r["id"], user_id=r["user_id"], kind=r["kind"], content=r["content"],
                   created_at=r["created_at"], last_seen_at=r["last_seen_at"],
                   access_count=r["access_count"], score=round(score, 6))
            for score, r in scored[:k]
        ]

    def touch(self, ids: list[str]) -> None:
        if not ids:
            return
        conn = self._connect()
        conn.executemany("UPDATE memories SET access_count = access_count + 1 WHERE id = ?", [(i,) for i in ids])
        conn.commit()
        conn.close()


def _select_store():
    from db.database import pg_available

    if pg_available():
        return PostgresSemanticStore()
    return SqliteSemanticStore()


class SemanticMemory:
    """Write-policy + embedding + retrieval over the selected store."""

    def __init__(self, embedder=None, store=None) -> None:
        from waagent.embeddings import get_embedder

        self.embedder = embedder or get_embedder()
        self.store = store or _select_store()

    def remember(self, user_id: str, text: str, source_msg_ids: Optional[list[str]] = None) -> Optional[Memory]:
        kind = classify_memory(text)
        if kind is None:
            return None
        content = strip_pii(text)
        mem = Memory(id=uuid.uuid4().hex, user_id=user_id, kind=kind, content=content)
        self.store.add(mem, self.embedder.embed(content))
        return mem

    def recall(self, user_id: str, text: str, k: Optional[int] = None) -> list[Memory]:
        query = self.embedder.embed(strip_pii(text))
        mems = self.store.search(user_id, query, k or settings.memory_top_k)
        # drop near-zero matches so unrelated turns are not "remembered"
        mems = [m for m in mems if (m.score is None or m.score >= 0.15)]
        if mems:
            self.store.touch([m.id for m in mems])
        return mems
