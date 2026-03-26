"""Shared utilities for the Korean Movie Recommender pipeline."""

import threading
import time


class TokenBucket:
    """Thread-safe token bucket rate limiter.

    Allows up to ``rate`` requests per ``window`` seconds. Each call to
    :meth:`acquire` blocks until a token is available, letting multiple
    threads share a single global rate limit safely.

    Args:
        rate: Maximum number of tokens per window.
        window: Length of the refill window in seconds.
    """

    def __init__(self, rate: int, window: float) -> None:
        self._rate = rate
        self._window = window
        self._tokens = float(rate)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available, then consume it."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._rate,
                    self._tokens + elapsed * (self._rate / self._window),
                )
                self._last_refill = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
            time.sleep(0.05)
