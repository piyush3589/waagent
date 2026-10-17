"""Worker: runs queued conversations off the request path.

With Redis up it consumes the shared queue (`python -m waagent.worker`);
offline the API process runs an in-process thread queue instead.
"""
from __future__ import annotations

import argparse

from waagent.agent import ConversationAgent
from waagent.logging_utils import get_logger, log_event
from waagent.queue import JobQueue

log = get_logger("waagent.worker")
_agent: ConversationAgent | None = None


def get_agent() -> ConversationAgent:
    global _agent
    if _agent is None:
        _agent = ConversationAgent()
    return _agent


def process_job(payload: dict) -> dict:
    """Payload: {user_id, text, request_id?} -> the reply as a plain dict."""
    result = get_agent().run(
        payload["user_id"], payload["text"], request_id=payload.get("request_id", "")
    )
    log_event(log, "job_done", user_id=payload["user_id"], intent=result.reply.intent.value,
              ms=result.reply.latency_ms)
    return result.reply.model_dump(mode="json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="wa-agent worker")
    parser.add_argument("--once", action="store_true", help="drain the local queue and exit")
    args = parser.parse_args(argv)

    q = JobQueue(handler=process_job)
    if args.once:
        processed = 0
        while q.drain_once(process_job, timeout=0.2):
            processed += 1
        print(f"processed {processed} job(s)")
        return
    if q.backend == "redis":
        print("worker: consuming redis queue")
        q.consume_forever()
    else:
        print("worker: no redis - the API process handles jobs on an in-process thread")


if __name__ == "__main__":
    main()
