from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class AcquireResult:
    acquired: bool
    reason: Optional[str] = None


class BusyDetector:
    """
    Thread-safe busy detector using a bounded semaphore.
    - acquire_nowait: if cannot acquire immediately => busy
    """
    def __init__(self, max_concurrent: int = 1) -> None:
        self.max_concurrent = max(1, int(max_concurrent))
        self._sem = threading.BoundedSemaphore(self.max_concurrent)

    def acquire_nowait(self) -> AcquireResult:
        ok = self._sem.acquire(blocking=False)
        if not ok:
            return AcquireResult(acquired=False, reason="local_busy")
        return AcquireResult(acquired=True)

    def release(self) -> None:
        # release can throw ValueError if released too much; ignore to be safe
        try:
            self._sem.release()
        except ValueError:
            pass


class CircuitBreaker:
    """
    Minimal circuit breaker for local provider:
    - after N consecutive failures => open for reset_sec
    """
    def __init__(self, fails_to_open: int = 3, reset_sec: int = 60) -> None:
        self.fails_to_open = max(1, int(fails_to_open))
        self.reset_sec = max(1, int(reset_sec))
        self._lock = threading.Lock()
        self._fails = 0
        self._open_until = 0.0

    def is_open(self) -> bool:
        with self._lock:
            now = time.time()
            if now < self._open_until:
                return True
            if self._open_until != 0.0 and now >= self._open_until:
                # auto close
                self._open_until = 0.0
                self._fails = 0
            return False

    def record_success(self) -> None:
        with self._lock:
            self._fails = 0
            self._open_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._fails += 1
            if self._fails >= self.fails_to_open:
                self._open_until = time.time() + self.reset_sec
