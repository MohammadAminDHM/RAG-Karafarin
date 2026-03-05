from __future__ import annotations
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Adds X-Process-Time-Ms header.
    """
    header_name = "X-Process-Time-Ms"

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response.headers[self.header_name] = str(elapsed_ms)
        # Make it accessible to handlers as well
        request.state.process_time_ms = elapsed_ms
        return response
