"""Scorecard: run golden conversations WITH and WITHOUT long-term memory.

The headline number is the memory delta: identical conversations scored under
memory ON vs memory OFF, plus the token-cost delta. Writes
eval/results/latest.json and prints a scorecard.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from waagent.config import settings  # noqa: E402

# Isolate eval storage from the app's dev databases and force offline backends.
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
settings.sqlite_path = str(RESULTS / "eval_store.db")
settings.redis_url = "redis://localhost:6390/0"  # nothing listens -> dict fallback
settings.database_url = "postgresql+psycopg2://waagent:waagent@localhost:6543/waagent"  # -> sqlite

from waagent.agent import ConversationAgent  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden_conversations.jsonl"
_LONG_DIGITS = re.compile(r"\d{8,}")
_RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def check(result, expect: dict) -> list[str]:
    failures: list[str] = []
    reply = result.reply
    if "intent" in expect and reply.intent.value != expect["intent"]:
        failures.append(f"intent={reply.intent.value} != {expect['intent']}")
    if "tool" in expect and expect["tool"] not in reply.tools_used:
        failures.append(f"missing tool {expect['tool']} (used {reply.tools_used})")
    if "handoff" in expect and reply.handoff != expect["handoff"]:
        failures.append(f"handoff={reply.handoff} != {expect['handoff']}")
    if "recall_min" in expect and len(result.retrieved) < expect["recall_min"]:
        failures.append(f"recalled {len(result.retrieved)} < {expect['recall_min']}")
    if "written_min" in expect and len(result.written) < expect["written_min"]:
        failures.append(f"wrote {len(result.written)} < {expect['written_min']}")
    if "confidence_min" in expect and reply.confidence < expect["confidence_min"]:
        failures.append(f"confidence {reply.confidence} < {expect['confidence_min']}")
    for needle in expect.get("reply_contains", []):
        if needle.lower() not in reply.text.lower():
            failures.append(f"reply missing '{needle}'")
    for needle in expect.get("reply_not_contains", []):
        if needle.lower() in reply.text.lower():
            failures.append(f"reply should not contain '{needle}'")
    if expect.get("written_no_pii"):
        for mem in result.written:
            if "@" in mem.content or _LONG_DIGITS.search(mem.content):
                failures.append(f"PII leaked into memory: {mem.content!r}")
    return failures


def run_conversation(agent: ConversationAgent, conv: dict, memory_enabled: bool) -> tuple[list[str], int]:
    uid = f"{conv['id']}-{'on' if memory_enabled else 'off'}-{_RUN_TS}"
    failures: list[str] = []
    tokens = 0
    for i, turn in enumerate(conv["turns"]):
        result = agent.run(uid, turn["user"], request_id=f"{conv['id']}-{i}")
        tokens += result.tokens
        failures += [f"turn{i}: {f}" for f in check(result, turn.get("expect", {}))]
    return failures, tokens


def main() -> None:
    store = RESULTS / "eval_store.db"
    if store.exists():
        store.unlink()

    conversations = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    report = {"memory_on": {}, "memory_off": {}, "by_category": {}, "failures": {"on": [], "off": []}}
    scores: dict[str, dict] = {"on": defaultdict(lambda: [0, 0]), "off": defaultdict(lambda: [0, 0])}
    tokens = {"on": 0, "off": 0}

    # One agent per mode, reused across conversations (unique user ids isolate memory).
    agents = {"on": ConversationAgent(memory_enabled=True), "off": ConversationAgent(memory_enabled=False)}

    for conv in conversations:
        category = conv.get("category", "uncategorised")
        for mode, enabled in (("on", True), ("off", False)):
            failures, tok = run_conversation(agents[mode], conv, enabled)
            tokens[mode] += tok
            passed = not failures
            scores[mode][category][1] += 1
            scores[mode]["ALL"][1] += 1
            if passed:
                scores[mode][category][0] += 1
                scores[mode]["ALL"][0] += 1
            else:
                report["failures"][mode].append({"id": conv["id"], "category": category, "reasons": failures})

    def pct(pair):
        return round(100.0 * pair[0] / pair[1], 1) if pair[1] else 0.0

    report["memory_on"] = {"passed": scores["on"]["ALL"][0], "total": scores["on"]["ALL"][1],
                           "accuracy_pct": pct(scores["on"]["ALL"]), "tokens": tokens["on"]}
    report["memory_off"] = {"passed": scores["off"]["ALL"][0], "total": scores["off"]["ALL"][1],
                            "accuracy_pct": pct(scores["off"]["ALL"]), "tokens": tokens["off"]}
    report["memory_delta_pct"] = round(report["memory_on"]["accuracy_pct"] - report["memory_off"]["accuracy_pct"], 1)
    report["by_category"] = {
        cat: {"on": f"{scores['on'][cat][0]}/{scores['on'][cat][1]}",
              "off": f"{scores['off'][cat][0]}/{scores['off'][cat][1]}"}
        for cat in sorted(k for k in scores["on"] if k != "ALL")
    }
    report["generated_at"] = _RUN_TS

    out = RESULTS / "latest.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    on, off = report["memory_on"], report["memory_off"]
    print("=" * 70)
    print(f"KREDIT-STYLE SCORECARD  (generated {_RUN_TS})")
    print("=" * 70)
    print(f"OVERALL (memory ON) : {on['passed']}/{on['total']}  ({on['accuracy_pct']}%)   tokens={on['tokens']}")
    print(f"OVERALL (memory OFF): {off['passed']}/{off['total']}  ({off['accuracy_pct']}%)   tokens={off['tokens']}")
    print(f"MEMORY DELTA        : +{report['memory_delta_pct']}% accuracy "
          f"(token cost +{on['tokens'] - off['tokens']} to remember/retrieve)")
    print("-" * 70)
    print("By category            memory ON   memory OFF")
    for cat, vals in report["by_category"].items():
        print(f"  {cat:<20} {vals['on']:>9} {vals['off']:>10}")
    print("-" * 70)
    if report["failures"]["on"]:
        print("Failures with memory ON:")
        for f in report["failures"]["on"]:
            print(f"  - {f['id']} [{f['category']}]: {', '.join(f['reasons'])}")
    print(f"report written -> {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
