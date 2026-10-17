"""Per-user rate limiting (Redis fixed-window, in-process fallback)."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from waagent.config import settings


class RateLimiter:
    def __init__(self, per_minute: int | None = None) -> None:
        self.per_minute = per_minute or settings.rate_limit_per_minute
        self._redis = None
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        try:
            import redis

            client = redis.Redis.from_url(
                settings.redis_url, decode_responses=True,
                socket_connect_timeout=1, socket_timeout=1,
            )
            client.ping()
            self._redis = client
        except Exception:
            self._redis = None

    @property
    def backend(self) -> str:
        return "redis" if self._redis else "dict"

    def allow(self, key: str) -> bool:
        if self._redis:
            bucket = f"rl:{key}:{int(time.time() // 60)}"
            count = self._redis.incr(bucket)
            if count == 1:
                self._redis.expire(bucket, 70)
            return count <= self.per_minute
        now = time.time()
        with self._lock:
            window = self._hits[key]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.per_minute:
                return False
            window.append(now)
            return True
