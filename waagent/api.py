"""FastAPI surface: WhatsApp webhook + local simulator + health + metrics.

The webhook is Meta-shaped (entry -> changes -> value -> messages) and verifies
the X-Hub-Signature-256 HMAC when WA_APP_SECRET is set, so real Cloud API
credentials drop straight in. Without them, POST /chat is the local simulator.
"""
from __future__ import annotations

import hashlib
import hmac
import uuid

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from waagent.agent import ConversationAgent
from waagent.config import settings
from waagent.logging_utils import get_logger, log_event
from waagent.metrics import metrics
from waagent.queue import JobQueue
from waagent.ratelimit import RateLimiter
from waagent.worker import process_job

log = get_logger("waagent.api")

app = FastAPI(title="wa-agent", version="0.1.0")

_agent: ConversationAgent | None = None
_limiter: RateLimiter | None = None
_queue: JobQueue | None = None


def get_agent() -> ConversationAgent:
    global _agent
    if _agent is None:
        _agent = ConversationAgent()
        log_event(log, "agent_ready", session=_agent.session.backend,
                  semantic=_agent.semantic.store.backend if _agent.semantic else "off",
                  llm=_agent.llm.name)
    return _agent


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter


def get_queue() -> JobQueue:
    global _queue
    if _queue is None:
        _queue = JobQueue(handler=process_job)
    return _queue


class ChatIn(BaseModel):
    user_id: str = "sim-user"
    text: str


class ChatOut(BaseModel):
    user_id: str
    reply: str
    intent: str
    confidence: float
    handoff: bool
    tools_used: list[str]
    memories_used: int
    request_id: str
    latency_ms: int


def _extract_messages(payload: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value", {}) or {}
            for msg in value.get("messages", []) or []:
                sender = msg.get("from")
                text = (msg.get("text") or {}).get("body")
                if sender and text:
                    out.append((str(sender), str(text)))
    return out


def _valid_signature(raw: bytes, header: str | None) -> bool:
    if not settings.app_secret:
        return True  # signature checks disabled in local/dev mode
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(settings.app_secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.split("=", 1)[1])


def _handle(user_id: str, text: str) -> ChatOut:
    limiter = get_limiter()
    if not limiter.allow(user_id):
        metrics.incr("rate_limited")
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    request_id = uuid.uuid4().hex
    metrics.incr("requests")
    result = get_agent().run(user_id, text, request_id=request_id)
    metrics.observe(result.reply.latency_ms)
    if result.reply.handoff:
        metrics.incr("handoffs")
    if result.retrieved:
        metrics.incr("memory_hits")

    log_event(log, "turn", request_id=request_id, user_id=user_id,
              intent=result.reply.intent.value, confidence=result.reply.confidence,
              handoff=result.reply.handoff, tools=result.reply.tools_used,
              memories=len(result.retrieved), ms=result.reply.latency_ms)

    return ChatOut(
        user_id=user_id,
        reply=result.reply.text,
        intent=result.reply.intent.value,
        confidence=result.reply.confidence,
        handoff=result.reply.handoff,
        tools_used=result.reply.tools_used,
        memories_used=len(result.retrieved),
        request_id=request_id,
        latency_ms=result.reply.latency_ms,
    )


@app.get("/healthz")
def healthz() -> dict:
    agent = get_agent()
    return {
        "status": "ok",
        "llm": agent.llm.name,
        "session_backend": agent.session.backend,
        "semantic_backend": agent.semantic.store.backend if agent.semantic else "off",
        "ratelimit_backend": get_limiter().backend,
    }


@app.get("/metrics")
def get_metrics() -> dict:
    return metrics.snapshot()


@app.get("/webhook/whatsapp")
def verify_webhook(
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
) -> PlainTextResponse:
    """Meta webhook verification handshake."""
    if hub_mode == "subscribe" and hub_verify_token == settings.verify_token:
        return PlainTextResponse(content=hub_challenge, status_code=200)
    raise HTTPException(status_code=403, detail="verification failed")


@app.post("/webhook/whatsapp", status_code=202)
async def inbound(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
) -> dict:
    raw = await request.body()
    if not _valid_signature(raw, x_hub_signature_256):
        metrics.incr("bad_signature")
        raise HTTPException(status_code=401, detail="invalid signature")

    payload = await request.json()
    limiter = get_limiter()
    queue = get_queue()
    jobs = []
    for user_id, text in _extract_messages(payload):
        request_id = uuid.uuid4().hex
        if not limiter.allow(user_id):
            metrics.incr("rate_limited")
            jobs.append({"user_id": user_id, "status": "rate_limited"})
            continue
        metrics.incr("accepted")
        job_id = queue.enqueue({"user_id": user_id, "text": text, "request_id": request_id})
        log_event(log, "enqueued", job_id=job_id, request_id=request_id, user_id=user_id)
        jobs.append({"user_id": user_id, "job_id": job_id, "status": "accepted"})
    return {"status": "accepted", "jobs": jobs}


@app.post("/chat/async", status_code=202)
def chat_async(body: ChatIn) -> dict:
    """Simulator variant that goes through the queue (off the request path)."""
    limiter = get_limiter()
    if not limiter.allow(body.user_id):
        metrics.incr("rate_limited")
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    request_id = uuid.uuid4().hex
    job_id = get_queue().enqueue({"user_id": body.user_id, "text": body.text, "request_id": request_id})
    metrics.incr("accepted")
    return {"job_id": job_id, "status": "queued", "request_id": request_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = get_queue().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    """Local simulator endpoint - no Meta signature, synchronous reply."""
    return _handle(body.user_id, body.text)
