"""Tool integrations: a typed, read-only registry the agent may call.

Each tool = a pydantic input model + a handler. The identical registry is
exposed over MCP in mcp/server.py, so the agent and any MCP client share one
source of truth. v1 is read-only; escalate_to_human is a stub the handoff
node acts on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from waagent.config import settings


class ToolError(Exception):
    """Raised by a tool for an expected, user-facing failure."""


class GetSessionInfoIn(BaseModel):
    user_id: str


class SearchFaqIn(BaseModel):
    query: str
    limit: int = 3


class GetOrderStatusIn(BaseModel):
    order_id: str


class CheckRefundPolicyIn(BaseModel):
    product: str = "default"


class EscalateToHumanIn(BaseModel):
    reason: str


_FAQ = [
    {"q": "how do I track my order", "a": "Open Orders, tap the order and you will see live tracking."},
    {"q": "what is your refund policy", "a": "Full refund within 7 days of delivery, or store credit up to 14 days."},
    {"q": "how to change delivery address", "a": "Open the order and choose Change address before it ships."},
    {"q": "do you offer cash on delivery", "a": "COD is available for orders under 5000 in most pin codes."},
    {"q": "how to contact support", "a": "Reply 'human' here and an agent will pick up, or email support@example.com."},
    {"q": "how long does delivery take", "a": "Metro cities 2-3 days, rest of India 4-6 business days."},
]

_ORDERS = {
    "1234": {"item": "Wireless Earbuds", "status": "packed", "eta": "2 days"},
    "5678": {"item": "Cotton Kurta", "status": "out_for_delivery", "eta": "today"},
    "9999": {"item": "Phone Case", "status": "delayed", "eta": "unknown"},
}


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "and", "or", "do", "does", "did",
    "i", "my", "me", "you", "your", "we", "our", "it", "this", "that",
    "what", "how", "when", "where", "which", "who", "why", "can", "could",
    "will", "would", "should", "please", "here", "there",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


def _overlap(a: str, b: str) -> int:
    return len(_tokens(a) & _tokens(b))


def search_faq(payload: SearchFaqIn, ctx: dict) -> dict:
    ranked = sorted(_FAQ, key=lambda f: _overlap(payload.query, f["q"]), reverse=True)
    matches = [
        f for f in ranked if _overlap(payload.query, f["q"]) >= settings.faq_min_overlap
    ][: payload.limit]
    return {"matches": matches}


def get_order_status(payload: GetOrderStatusIn, ctx: dict) -> dict:
    order = _ORDERS.get(payload.order_id)
    if order is None:
        raise ToolError(f"I could not find order {payload.order_id}.")
    return {"order_id": payload.order_id, **order}


def check_refund_policy(payload: CheckRefundPolicyIn, ctx: dict) -> dict:
    return {
        "policy": (
            "Refunds are free within 7 days of delivery. Between 7 and 14 days we "
            "offer store credit. After 14 days the item must be defective to qualify."
        )
    }


def get_session_info(payload: GetSessionInfoIn, ctx: dict) -> dict:
    return {"language": ctx.get("language", "en"), "handoff": bool(ctx.get("handoff", False))}


def escalate_to_human(payload: EscalateToHumanIn, ctx: dict) -> dict:
    return {"ack": f"Queued for a human agent. Reason: {payload.reason}"}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any, dict], dict]
    read_only: bool = True


REGISTRY: dict[str, ToolSpec] = {
    "search_faq": ToolSpec("search_faq", "Search the help-centre FAQ for an answer.", SearchFaqIn, search_faq),
    "get_order_status": ToolSpec("get_order_status", "Look up an order's status and ETA by order id.", GetOrderStatusIn, get_order_status),
    "check_refund_policy": ToolSpec("check_refund_policy", "Return the current refund policy.", CheckRefundPolicyIn, check_refund_policy),
    "get_session_info": ToolSpec("get_session_info", "Return session flags such as language.", GetSessionInfoIn, get_session_info),
    "escalate_to_human": ToolSpec("escalate_to_human", "Queue the conversation for a human agent.", EscalateToHumanIn, escalate_to_human),
}


def list_tools() -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "input": t.input_model.__name__}
        for t in REGISTRY.values()
    ]


def run_tool(name: str, payload: dict, ctx: dict | None = None) -> dict:
    """Validate input against the tool's model, run it, and never raise."""
    ctx = ctx or {}
    spec = REGISTRY.get(name)
    if spec is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        args = spec.input_model(**payload)
        return spec.handler(args, ctx)
    except ToolError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # bad input, etc.
        return {"error": f"{name} failed: {exc}"}
