"""Bounded, process-local request throttling for a single API worker."""

import time
from collections import deque

from fastapi import HTTPException


class WindowLimiter:
    def __init__(self, limit: int, max_clients: int = 10000):
        self.limit = limit
        self.max_clients = max_clients
        self.windows: dict[str, deque[float]] = {}

    def check(self, identity: str) -> None:
        now = time.monotonic()
        for client in list(self.windows):
            if now - self.windows[client][-1] >= 60:
                del self.windows[client]
        window = self.windows.get(identity)
        if window is None:
            if len(self.windows) >= self.max_clients:
                raise HTTPException(429, "rate_limit_exceeded", headers={"Retry-After": "60"})
            window = self.windows[identity] = deque()
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= self.limit:
            raise HTTPException(429, "rate_limit_exceeded", headers={"Retry-After": "60"})
        window.append(now)
