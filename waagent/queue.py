"""Job queue: a Redis list when available, an in-process thread queue otherwise.

Webhook ingestion enqueues a job and returns immediately; a worker (a separate
process when Redis is up, or a daemon thread offline) runs the agent off the
request path. Job status is pollable via GET /jobs/{id}.
"""
from __future__ import annotations

import json
import queue
import threading
import uuid
from typing import Callable, Optional

from waagent.config import settings


class JobQueue:
    def __init__(self, handler: Optional[Callable[[dict], dict]] = None, name: str = "wa:jobs",
                 run_thread: bool = True) -> None:
        self._name = name
        self._handler = handler
        self._results: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._redis = self._connect()
        self._thread: Optional[threading.Thread] = None
        if self._redis is None and handler is not None and run_thread:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def _connect(self):
        try:
            import redis

            client = redis.Redis.from_url(
                settings.redis_url, decode_responses=True,
                socket_connect_timeout=1, socket_timeout=1,
            )
            client.ping()
            return client
        except Exception:
            return None

    @property
    def backend(self) -> str:
        return "redis" if self._redis else "thread"

    def enqueue(self, payload: dict) -> str:
        job_id = uuid.uuid4().hex
        if self._redis is not None:
            self._redis.hset(f"job:{job_id}", mapping={"status": "queued"})
            self._redis.lpush(self._name, json.dumps({"id": job_id, "payload": payload}))
        else:
            job = {"id": job_id, "payload": payload, "status": "queued"}
            with self._lock:
                self._results[job_id] = job
            self._q.put(job)
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        if self._redis is not None:
            data = self._redis.hgetall(f"job:{job_id}")
            if not data:
                return None
            if data.get("result"):
                data["result"] = json.loads(data["result"])
            return data
        with self._lock:
            job = self._results.get(job_id)
            return dict(job) if job else None

    def _process(self, job: dict, handler: Optional[Callable[[dict], dict]] = None) -> None:
        fn = handler or self._handler
        try:
            job["result"] = fn(job["payload"]) if fn else None
            job["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            job["status"] = "error"
            job["result"] = {"error": str(exc)}
        if self._redis is not None:
            self._redis.hset(
                f"job:{job['id']}",
                mapping={"status": job["status"], "result": json.dumps(job.get("result"))},
            )
        else:
            with self._lock:
                self._results[job["id"]] = job

    def _run_loop(self) -> None:
        while True:
            job = self._q.get()
            self._process(job)

    def drain_once(self, handler: Optional[Callable[[dict], dict]] = None, timeout: float = 1.0) -> bool:
        """Process one queued job synchronously (worker --once, tests)."""
        try:
            job = self._q.get(timeout=timeout)
        except queue.Empty:
            return False
        self._process(job, handler)
        return True

    def consume_forever(self) -> None:
        """Blocking Redis worker loop."""
        assert self._redis is not None
        while True:
            item = self._redis.brpop(self._name, timeout=5)
            if not item:
                continue
            job = json.loads(item[1])
            self._process(job)
