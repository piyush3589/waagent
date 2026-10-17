-- wa-agent initial schema (Postgres 16 + pgvector).
-- Mounted into /docker-entrypoint-initdb.d so a fresh volume bootstraps itself.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS conversations (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT        NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    handoff     BOOLEAN     NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS ix_conversations_user ON conversations (user_id);

CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT        NOT NULL,
    role        TEXT        NOT NULL,          -- user | assistant
    content     TEXT        NOT NULL,
    request_id  TEXT,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_messages_user_ts ON messages (user_id, ts);

CREATE TABLE IF NOT EXISTS memories (
    id             TEXT        PRIMARY KEY,
    user_id        TEXT        NOT NULL,
    kind           TEXT        NOT NULL,       -- preference | fact | event | sentiment
    content        TEXT        NOT NULL,
    embedding      vector(256) NOT NULL,
    source_msg_ids TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    access_count   INTEGER     NOT NULL DEFAULT 0,
    archived       BOOLEAN     NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS ix_memories_user_active
    ON memories (user_id) WHERE archived = false;
-- ANN index for cosine distance (pgvector >= 0.5)
CREATE INDEX IF NOT EXISTS ix_memories_embedding
    ON memories USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS traces (
    id         BIGSERIAL PRIMARY KEY,
    request_id TEXT        NOT NULL,
    user_id    TEXT,
    node       TEXT        NOT NULL,
    ms         INTEGER     NOT NULL DEFAULT 0,
    tokens     INTEGER     NOT NULL DEFAULT 0,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_traces_request ON traces (request_id);
