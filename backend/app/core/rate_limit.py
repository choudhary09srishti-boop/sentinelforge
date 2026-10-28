"""In-memory sliding-window rate limiting, keyed by client address.

Two buckets per client: all requests, and a smaller one for expensive POSTs (uploads, indexing,
starting an agent run). Limits are counted before authentication so key guessing is throttled
too. State lives in this process only, which is enough for a single-instance prototype.
"""

import math
import time
from collections import deque

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.errors import error_response

WINDOW_SECONDS = 60.0
MAX_TRACKED_KEYS = 10_000
EXEMPT_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})
HEAVY_POST_PATHS = frozenset(
    {"/api/v1/agent/runs", "/api/v1/documents", "/api/v1/documents/index-workspace"}
)


class RateLimitSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    rate_limit_per_minute: int = Field(default=120, ge=1, le=100_000)
    rate_limit_heavy_per_minute: int = Field(default=20, ge=1, le=100_000)
    # Only enable behind a proxy you control (for example Render): the header is client-supplied.
    rate_limit_trust_proxy: bool = False


class SlidingWindow:
    def __init__(self, limit: int, window: float = WINDOW_SECONDS) -> None:
        self._limit = limit
        self._window = window
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str, now: float) -> int:
        """Record a hit. Returns 0 when allowed, else the seconds until a slot frees up."""
        if key not in self._hits and len(self._hits) >= MAX_TRACKED_KEYS:
            self._prune(now)
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= self._limit:
            return max(1, math.ceil(self._window - (now - hits[0])))
        hits.append(now)
        return 0

    def _prune(self, now: float) -> None:
        stale = [k for k, v in self._hits.items() if not v or now - v[-1] >= self._window]
        for key in stale:
            del self._hits[key]
        if len(self._hits) >= MAX_TRACKED_KEYS:
            self._hits.clear()


def client_key(request: Request, trust_proxy: bool) -> str:
    if trust_proxy:
        first = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if first:
            return first[:64]
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, config: RateLimitSettings) -> None:
        super().__init__(app)
        self._general = SlidingWindow(config.rate_limit_per_minute)
        self._heavy = SlidingWindow(config.rate_limit_heavy_per_minute)
        self._trust_proxy = config.rate_limit_trust_proxy

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if request.method == "OPTIONS" or path in EXEMPT_PATHS:
            return await call_next(request)
        key = client_key(request, self._trust_proxy)
        now = time.monotonic()
        wait = self._general.check(key, now)
        if wait == 0 and request.method == "POST" and path in HEAVY_POST_PATHS:
            wait = self._heavy.check(key, now)
        if wait:
            message = "Too many requests. Try again shortly."
            response = error_response(request, 429, "rate_limited", message)
            response.headers["Retry-After"] = str(wait)
            return response
        return await call_next(request)
