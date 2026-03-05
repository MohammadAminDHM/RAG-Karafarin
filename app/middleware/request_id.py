from __future__ import annotations
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Ensures every request has a request_id and exposes it in response headers.
    - Accepts incoming X-Request-ID if present
    - Otherwise generates a UUID4
    """
    header_name = "X-Request-ID"

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get(self.header_name)
        rid = incoming.strip() if incoming else str(uuid.uuid4())

        request.state.request_id = rid
        response: Response = await call_next(request)
        response.headers[self.header_name] = rid
        return response
