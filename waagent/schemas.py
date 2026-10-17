"""Pydantic contracts shared across the pipeline (the agent's typed surface)."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryKind(str, Enum):
    preference = "preference"
    fact = "fact"
    event = "event"
    sentiment = "sentiment"


class Intent(str, Enum):
    smalltalk = "smalltalk"
    faq = "faq"
    billing = "billing"
    support = "support"
    complaint = "complaint"
    off_topic = "off_topic"
    handoff_request = "handoff_request"


class Memory(BaseModel):
    """A single long-term, semantic fact worth remembering about a user."""

    id: str
    user_id: str
    kind: MemoryKind = MemoryKind.fact
    content: str
    created_at: datetime = Field(default_factory=utcnow)
    last_seen_at: datetime = Field(default_factory=utcnow)
    access_count: int = 0
    score: Optional[float] = None  # cosine similarity, set on retrieval


class Turn(BaseModel):
    user_id: str
    text: str
    role: str = "user"  # user | assistant
    ts: datetime = Field(default_factory=utcnow)


class Plan(BaseModel):
    intent: Intent = Intent.support
    use_tools: list[str] = Field(default_factory=list)
    needs_human: bool = False
    reason: str = ""


class Reply(BaseModel):
    user_id: str
    text: str
    intent: Intent = Intent.support
    confidence: float = 1.0
    handoff: bool = False
    tools_used: list[str] = Field(default_factory=list)
    memories_used: list[str] = Field(default_factory=list)
    request_id: str = ""
    latency_ms: int = 0


class AgentResult(BaseModel):
    """The full record of one agent turn (what the API and eval both consume)."""

    user_id: str
    question: str
    reply: Reply
    plan: Plan
    summary: str = ""
    retrieved: list[Memory] = Field(default_factory=list)
    written: list[Memory] = Field(default_factory=list)
    tokens: int = 0
