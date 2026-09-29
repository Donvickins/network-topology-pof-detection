import anyio
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Fails requests that run longer than timeout_s with a 504."""

    def __init__(self, app, timeout_s: int):
        super().__init__(app)
        self.timeout_s = timeout_s

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            with anyio.fail_after(self.timeout_s):
                return await call_next(request)
        except TimeoutError:
            return JSONResponse(
                status_code=504,
                content={'status': 'error', 'message': 'Request timed out'},
            )
