"""The agent loop: a LangGraph state graph.

    START -> planner -> retrieve -> tools -> responder --+--> handoff -> END
                                                          |
                                                          +--> END

- planner   : classify intent + pick tools (LLM, grounded in the tool list)
- retrieve  : pull short-term summary + top-k long-term memories
- tools     : execute the selected read-only tools (typed, never raises)
- responder : compose a grounded reply + confidence
- handoff   : interrupt() for a human when intent/confidence requires it
"""
from __future__ import annotations

import re
import time
from typing import Any, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from waagent.config import settings
from waagent.llm import LLMClient, get_llm
from waagent.memory import (
    SemanticMemory,
    SessionStore,
    compact,
    needs_compaction,
)
from waagent.schemas import AgentResult, Intent, Memory, Plan, Reply
from waagent.tools import run_tool

try:  # pragma: no cover - import shape varies by langgraph version
    from langgraph.errors import GraphInterrupt
except Exception:  # noqa: BLE001
    class GraphInterrupt(Exception):  # type: ignore[no-redef]
        pass

_ORDER_ID = re.compile(r"\b(\d{3,})\b")


class AgentState(TypedDict, total=False):
    user_id: str
    question: str
    request_id: str
    history: list[dict]
    summary: str
    plan: dict
    retrieved: list[dict]
    tool_results: list[dict]
    tools_used: list[str]
    reply_text: str
    confidence: float
    needs_handoff: bool
    handoff: bool
    tokens: int


class ConversationAgent:
    def __init__(
        self,
        session: Optional[SessionStore] = None,
        semantic: Optional[SemanticMemory] = None,
        llm: Optional[LLMClient] = None,
        memory_enabled: bool = True,
        use_interrupt: bool = False,
    ) -> None:
        self.session = session or SessionStore()
        self.semantic = semantic if semantic is not None else (SemanticMemory() if memory_enabled else None)
        self.llm = llm or get_llm()
        self.use_interrupt = use_interrupt
        self.checkpointer = MemorySaver()
        self.graph = self._build()

    # ------------------------------------------------------------------ #
    # graph
    # ------------------------------------------------------------------ #
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("planner", self._planner)
        g.add_node("retrieve", self._retrieve)
        g.add_node("tools", self._tools)
        g.add_node("responder", self._responder)
        g.add_node("human_handoff", self._handoff)
        g.set_entry_point("planner")
        g.add_edge("planner", "retrieve")
        g.add_edge("retrieve", "tools")
        g.add_edge("tools", "responder")
        g.add_conditional_edges("responder", self._route, {"handoff": "human_handoff", "end": END})
        g.add_edge("human_handoff", END)
        return g.compile(checkpointer=self.checkpointer)

    def _route(self, state: AgentState) -> str:
        return "handoff" if state.get("needs_handoff") else "end"

    # ------------------------------------------------------------------ #
    # nodes
    # ------------------------------------------------------------------ #
    def _planner(self, state: AgentState) -> dict:
        plan = self.llm.plan(state["question"], state.get("history", []), state.get("summary", ""))
        return {"plan": plan.model_dump(mode="json"), "tokens": state.get("tokens", 0) + 20}

    def _retrieve(self, state: AgentState) -> dict:
        user_id = state["user_id"]
        summary = self.session.get_summary(user_id)
        retrieved = []
        if self.semantic is not None:
            retrieved = [m.model_dump(mode="json") for m in self.semantic.recall(user_id, state["question"])]
        # token estimate grows with what we inject into context
        mem_tokens = 15 * len(retrieved) + len(summary) // 4
        return {"summary": summary, "retrieved": retrieved, "tokens": state.get("tokens", 0) + 10 + mem_tokens}

    def _tools(self, state: AgentState) -> dict:
        plan = Plan(**state["plan"])
        ctx = self.session.get_flags(state["user_id"])
        results, used = [], []
        for name in plan.use_tools:
            args = self._tool_args(name, state["question"], state["user_id"])
            results.append({"tool": name, "output": run_tool(name, args, ctx)})
            used.append(name)
        return {"tool_results": results, "tools_used": used}

    def _responder(self, state: AgentState) -> dict:
        plan = Plan(**state["plan"])
        memories = [Memory(**m) for m in state.get("retrieved", [])]
        text, confidence = self.llm.respond(
            state["question"], memories, state.get("tool_results", []), state.get("summary", ""), plan
        )
        needs_handoff = plan.needs_human or confidence < settings.min_reply_confidence
        return {
            "reply_text": text,
            "confidence": confidence,
            "needs_handoff": needs_handoff,
            "tokens": state.get("tokens", 0) + 30 + 20 * len(memories),
        }

    def _handoff(self, state: AgentState) -> dict:
        if self.use_interrupt:
            from langgraph.types import interrupt

            decision = interrupt({"reason": "handoff", "question": state["question"]})
            return {"handoff": True, "reply_text": str(decision)}
        return {
            "handoff": True,
            "reply_text": "I'm connecting you to a human agent who can take it from here. One moment please.",
        }

    @staticmethod
    def _tool_args(name: str, question: str, user_id: str) -> dict[str, Any]:
        if name == "get_order_status":
            match = _ORDER_ID.search(question)
            return {"order_id": match.group(1)} if match else {}
        if name == "search_faq":
            return {"query": question, "limit": 3}
        if name == "get_session_info":
            return {"user_id": user_id}
        if name == "escalate_to_human":
            return {"reason": question}
        return {}

    # ------------------------------------------------------------------ #
    # public entry point
    # ------------------------------------------------------------------ #
    def run(self, user_id: str, question: str, request_id: str = "") -> AgentResult:
        started = time.perf_counter()
        self.session.append_turn(user_id, "user", question)
        initial: AgentState = {
            "user_id": user_id,
            "question": question,
            "request_id": request_id,
            "history": self.session.get_turns(user_id),
            # Reset per-turn transient fields: the checkpointer keeps state per
            # thread_id, so without this the previous turn's handoff/plan leaks.
            "plan": {},
            "summary": "",
            "retrieved": [],
            "tool_results": [],
            "tools_used": [],
            "reply_text": "",
            "confidence": 0.0,
            "needs_handoff": False,
            "handoff": False,
            "tokens": 0,
        }
        config = {"configurable": {"thread_id": user_id}, "recursion_limit": settings.max_agent_iterations * 4}
        try:
            final = self.graph.invoke(initial, config)
        except GraphInterrupt:
            final = {
                **initial,
                "reply_text": "Connecting you to a human agent now.",
                "confidence": 0.0,
                "handoff": True,
                "needs_handoff": True,
                "plan": Plan(intent=Intent.handoff_request, needs_human=True).model_dump(mode="json"),
                "retrieved": [],
                "tool_results": [],
                "tools_used": [],
            }
        # invoke() returns the state paused at an interrupt. Detect a pending
        # human_handoff, then resume the graph with the handoff message so the
        # conversation thread completes (rather than staying paused).
        snapshot = self.graph.get_state(config)
        if "human_handoff" in (getattr(snapshot, "next", ()) or ()):
            from langgraph.types import Command

            final = self.graph.invoke(
                Command(resume="I'm connecting you to a human agent who can take it from here. One moment please."),
                config,
            )
        return self._finalise(user_id, question, request_id, final, started)

    def _finalise(
        self, user_id: str, question: str, request_id: str, final: dict, started: float
    ) -> AgentResult:
        reply_text = final.get("reply_text", "")
        self.session.append_turn(user_id, "assistant", reply_text)

        written: list[Memory] = []
        if self.semantic is not None:
            mem = self.semantic.remember(user_id, question)
            if mem is not None:
                written.append(mem)

        if needs_compaction(self.session, user_id):
            compact(self.session, user_id, self.llm.summarize)

        plan = Plan(**final.get("plan", {}))
        retrieved = [Memory(**m) for m in final.get("retrieved", [])]
        latency_ms = int((time.perf_counter() - started) * 1000)
        reply = Reply(
            user_id=user_id,
            text=reply_text,
            intent=plan.intent,
            confidence=float(final.get("confidence", 0.0)),
            handoff=bool(final.get("handoff", False)),
            tools_used=final.get("tools_used", []),
            memories_used=[m.id for m in retrieved],
            request_id=request_id,
            latency_ms=latency_ms,
        )
        return AgentResult(
            user_id=user_id,
            question=question,
            reply=reply,
            plan=plan,
            summary=final.get("summary", ""),
            retrieved=retrieved,
            written=written,
            tokens=int(final.get("tokens", 0)),
        )
