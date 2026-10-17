import os
import tempfile

# Force fully-offline, deterministic test mode before waagent.config is imported:
# no LLM, mock embeddings, an unreachable Postgres (so the SQLite store is used),
# and an unreachable Redis (so the in-process session dict is used).
os.environ.setdefault("WA_LLM_PROVIDER", "mock")
os.environ.setdefault("WA_EMBEDDER", "mock")
os.environ["WA_SQLITE_PATH"] = os.path.join(tempfile.mkdtemp(prefix="wa-test-"), "test.db")
os.environ["WA_DATABASE_URL"] = "postgresql+psycopg2://waagent:waagent@localhost:6543/waagent"
os.environ["WA_REDIS_URL"] = "redis://localhost:6390/0"
