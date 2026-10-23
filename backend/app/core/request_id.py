"""Attach a server-generated request ID to every request and response."""

import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response


def add_request_id_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Always generated here, never taken from the client, to avoid log injection.
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
