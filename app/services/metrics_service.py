from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import threading


@dataclass
class Metrics:
    lock: threading.Lock = field(default_factory=threading.Lock)
    counters: Dict[str, int] = field(default_factory=lambda: {
        "requests_total": 0,
        "requests_ok": 0,
        "requests_error": 0,
        "queries_total": 0,
        "reindex_total": 0,
    })

    def inc(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.counters[key] = int(self.counters.get(key, 0)) + int(n)

    def snapshot(self) -> Dict[str, int]:
        with self.lock:
            return dict(self.counters)
