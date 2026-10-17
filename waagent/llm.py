"""LLM clients.

MockLLM (default) is a deterministic rule engine so the whole pipeline runs
offline and the eval harness is reproducible. OpenAICompatibleLLM talks to any
OpenAI-style /chat/completions endpoint (Groq, OpenAI) when a key is present.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Protocol

from waagent.config import settings
from waagent.schemas import Intent, Memory, Plan
from waagent.tools import list_tools

_ORDER_ID = re.compile(r"\b(\d{3,})\b")


def _clean_list(items) -> list[str]:
    return [str(x) for x in items] if isinstance(items, list) else []


class LLMClient(Protocol):
    name: str

    def plan(self, question: str, history: list[dict], summary: str) -> Plan: ...

    def respond(self, question: str, memories: list[Memory], tool_results: list[dict], summary: str, plan: Plan) -> tuple[str, float]: ...

    def summarize(self, prior_summary: str, old_turns: list[dict]) -> str: ...


class MockLLM:
    """Deterministic stand-in. Same code path as a real model, no key needed."""

    name = "mock"

    def plan(self, question: str, history: list[dict], summary: str) -> Plan:
        q = question.lower()
        if any(k in q for k in ("human", "representative", "real person", "talk to someone", "agent")):
            return Plan(intent=Intent.handoff_request, needs_human=True, reason="user asked for a human")
        if any(k in q for k in ("refund", "money back")):
            return Plan(intent=Intent.billing, use_tools=["check_refund_policy"])
        if any(k in q for k in ("order", "delivery", "track", "ship")):
            if _ORDER_ID.search(q):
                return Plan(intent=Intent.support, use_tools=["get_order_status"])
            return Plan(intent=Intent.support, use_tools=["search_faq"])
        if q.startswith(("how", "what", "where", "when", "can i", "do you", "is there", "does")):
            return Plan(intent=Intent.faq, use_tools=["search_faq"])
        if re.search(r"\b(hi|hello|hey|good (morning|evening|afternoon))\b", q):
            return Plan(intent=Intent.smalltalk)
        return Plan(intent=Intent.support)

    def respond(
        self,
        question: str,
        memories: list[Memory],
        tool_results: list[dict],
        summary: str,
        plan: Plan,
    ) -> tuple[str, float]:
        parts: list[str] = []
        confidence = 0.55

        if memories:
            recalled = "; ".join(m.content for m in memories[:2])
            parts.append(f"(I remember: {recalled}.)")
            confidence += 0.15

        for tr in tool_results:
            name, out = tr.get("tool"), tr.get("output", {})
            if "error" in out:
                parts.append(f"Sorry - {out['error']}")
                confidence = 0.5
            elif name == "get_order_status":
                parts.append(f"Your {out.get('item', 'order')} is {out['status']} (ETA: {out.get('eta')}).")
                confidence = 0.9
            elif name == "check_refund_policy":
                parts.append(out["policy"])
                confidence = 0.85
            elif name == "search_faq" and out.get("matches"):
                parts.append(out["matches"][0]["a"])
                confidence = 0.8

        if not parts:
            if plan.intent == Intent.smalltalk:
                parts.append("Hi! How can I help you today? I can track orders, explain refunds, or answer FAQs.")
                confidence = 0.7
            elif not question.strip().endswith("?") and question.lower().startswith(("i ", "my ", "we ", "our ", "i'm")):
                parts.append("Got it - I'll keep that in mind for our conversation.")
                confidence = 0.75
            else:
                parts.append("I'm not fully sure about that - let me get a human to help if you'd like.")
                confidence = 0.4

        return " ".join(parts), round(min(confidence, 0.95), 2)

    def summarize(self, prior_summary: str, old_turns: list[dict]) -> str:
        body = "; ".join(f"{t.get('role', 'user')}: {t.get('text', '')}" for t in old_turns)
        return f"{prior_summary} {body}".strip()[:800]


class OpenAICompatibleLLM:  # pragma: no cover - needs network + key
    """Groq / OpenAI via the shared /chat/completions shape."""

    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _chat(self, system: str, user: str) -> str:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "temperature": 0.2,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def plan(self, question: str, history: list[dict], summary: str) -> Plan:
        tool_names = ", ".join(t["name"] for t in list_tools())
        system = (
            "You are a routing planner for a WhatsApp support agent. "
            f"Available tools: {tool_names}. "
            'Reply with JSON only: {"intent": one of smalltalk|faq|billing|support|complaint|off_topic|handoff_request, '
            '"use_tools": [names], "needs_human": bool, "reason": str}'
        )
        user = f"Summary so far: {summary}\n\nUser message: {question}"
        try:
            data = json.loads(self._chat(system, user))
            return Plan(**data)
        except Exception:
            return MockLLM().plan(question, history, summary)

    def respond(self, question, memories, tool_results, summary, plan) -> tuple[str, float]:
        facts = "\n".join(f"- {m.content}" for m in memories) or "(none)"
        tools = json.dumps(tool_results, default=str)
        system = (
            "You are a helpful, concise WhatsApp support agent. Only use the "
            "remembered facts and tool output provided; never invent numbers."
        )
        user = f"Known about user:\n{facts}\n\nTool output:\n{tools}\n\nUser: {question}"
        text = self._chat(system, user)
        return text, 0.8

    def summarize(self, prior_summary, old_turns) -> str:
        convo = "; ".join(f"{t['role']}: {t['text']}" for t in old_turns)
        system = "Summarise the chat for future context in 2-3 sentences. Keep names, preferences and open issues."
        return self._chat(system, f"Prior summary: {prior_summary}\n\nNew turns: {convo}")


def get_llm() -> LLMClient:
    provider = settings.llm_provider.lower()
    if provider == "groq" and settings.groq_api_key:
        return OpenAICompatibleLLM("https://api.groq.com/openai/v1", settings.groq_api_key, settings.llm_model)
    if provider == "openai" and settings.openai_api_key:
        return OpenAICompatibleLLM("https://api.openai.com/v1", settings.openai_api_key, settings.llm_model)
    return MockLLM()
