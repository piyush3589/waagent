"""wa-agent local chat simulator.

Examples
--------
    python run.py                        # interactive REPL
    python run.py --demo                 # scripted multi-turn demo (memory + handoff)
    python run.py --memory-off           # same agent with memory disabled (A/B)
    python run.py "where is my order 1234"  # one-shot
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from waagent.agent import ConversationAgent
from waagent.schemas import AgentResult

DEMO_TURNS = [
    "hi",
    "I prefer Hindi and WhatsApp for updates",
    "which language should you reply in for updates?",
    "where is my order 1234",
    "what is my refund policy",
    "my order 9999 is delayed and I am very frustrated",
    "I want to talk to a human",
]


def _print(result: AgentResult) -> None:
    r = result.reply
    print(f"\n  you : {result.question}")
    print(f"  bot : {r.text}")
    meta = f"intent={r.intent.value} confidence={r.confidence} handoff={r.handoff}"
    if r.tools_used:
        meta += f" tools={','.join(r.tools_used)}"
    if result.retrieved:
        meta += f" recalled={len(result.retrieved)}"
    if result.written:
        meta += f" wrote={len(result.written)}"
    print(f"  [{meta}  {r.latency_ms}ms]")


def main() -> None:
    # Real models can reply in any language/emoji; the Windows console defaults
    # to cp1252 and would crash on non-ASCII. Force UTF-8 on stdout/stderr.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="wa-agent WhatsApp-style chat simulator.")
    parser.add_argument("message", nargs="*", help="one-shot message")
    parser.add_argument("--user", default=None, help="user id / phone")
    parser.add_argument("--demo", action="store_true", help="run the scripted multi-turn demo")
    parser.add_argument("--memory-off", action="store_true", help="disable long-term memory (A/B)")
    parser.add_argument("--interrupt", action="store_true", help="enable interrupt-based handoff")
    parser.add_argument("--json", action="store_true", help="print raw JSON per turn")
    args = parser.parse_args()

    # A demo run gets a fresh user so it always starts from an empty memory.
    user_id = args.user or (f"demo-{uuid.uuid4().hex[:8]}" if args.demo else "demo-user")
    agent = ConversationAgent(
        memory_enabled=not args.memory_off,
        use_interrupt=args.interrupt,
    )
    print(f"store: session={agent.session.backend}, semantic="
          f"{agent.semantic.store.backend if agent.semantic else 'off'}, llm={agent.llm.name}, user={user_id}")

    def handle(text: str) -> None:
        result = agent.run(user_id, text)
        if args.json:
            print(json.dumps(result.reply.model_dump(mode="json"), indent=2))
        else:
            _print(result)

    if args.demo:
        for turn in DEMO_TURNS:
            handle(turn)
        return

    if args.message:
        handle(" ".join(args.message))
        return

    print("wa-agent simulator. Type a message, or 'exit'.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit", ""}:
            break
        handle(text)


if __name__ == "__main__":
    main()
