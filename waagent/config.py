"""Env-driven settings. All knobs live here; nothing reads os.environ directly."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WA_",
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    llm_provider: str = "mock"  # mock | groq | openai
    groq_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"

    # --- Embeddings --------------------------------------------------------
    embedder: str = "mock"  # mock | openai
    embed_dim: int = 256

    # --- Infra -------------------------------------------------------------
    database_url: str = "postgresql+psycopg2://waagent:waagent@localhost:5432/waagent"
    sqlite_path: str = str(DATA_DIR / "waagent.db")  # offline fallback store
    redis_url: str = "redis://localhost:6379/0"

    # --- WhatsApp webhook --------------------------------------------------
    app_secret: str = ""  # Meta app secret; empty disables signature checks
    verify_token: str = "wa-verify-token"

    # --- Behaviour ---------------------------------------------------------
    session_max_turns: int = 8
    summary_trigger_turns: int = 6
    memory_top_k: int = 5
    min_reply_confidence: float = 0.55
    faq_min_overlap: int = 1  # content-word overlap required to accept an FAQ answer
    max_agent_iterations: int = 5
    rate_limit_per_minute: int = 20
    log_level: str = "INFO"


settings = Settings()
DATA_DIR.mkdir(parents=True, exist_ok=True)
