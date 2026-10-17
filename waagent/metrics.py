"""In-process metrics: counters + latency percentiles exposed at /metrics."""
from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self, max_latency_samples: int = 5000) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = defaultdict(int)
        self.latencies: list[int] = []
        self._max = max_latency_samples

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def observe(self, ms: int) -> None:
        with self._lock:
            self.latencies.append(ms)
            if len(self.latencies) > self._max:
                self.latencies = self.latencies[-self._max:]

    @staticmethod
    def _percentile(sorted_values: list[int], pct: float) -> int:
        if not sorted_values:
            return 0
        idx = min(len(sorted_values) - 1, int(round(pct / 100 * (len(sorted_values) - 1))))
        return sorted_values[idx]

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self.latencies)
            counters = dict(self.counters)
        return {
            "counters": counters,
            "latency_ms": {
                "count": len(lat),
                "p50": self._percentile(lat, 50),
                "p95": self._percentile(lat, 95),
                "p99": self._percentile(lat, 99),
            },
        }


metrics = Metrics()
