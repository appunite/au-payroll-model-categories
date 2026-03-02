"""Logging utilities and middleware for request tracking."""

import json
import logging
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.config import LOG_PERFORMANCE

# Context variable to store request ID across async calls
request_id_context: ContextVar[str] = ContextVar("request_id", default="")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add unique request ID to each request.

    The request ID is:
    1. Extracted from X-Request-ID header if present
    2. Generated as UUID if not present
    3. Added to response headers
    4. Stored in context for logging
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in context for access in logging
        request_id_context.set(request_id)

        # Add to request state for access in route handlers
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add to response headers
        response.headers["X-Request-ID"] = request_id

        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to track request processing time."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logging.getLogger("timing")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip timing for health checks to reduce noise
        if request.url.path == "/health":
            return await call_next(request)

        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Add timing header
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

        # Log if performance logging enabled
        if LOG_PERFORMANCE:
            request_id = getattr(request.state, "request_id", "unknown")

            self.logger.info(
                f"request_id={request_id} method={request.method} "
                f"path={request.url.path} status={response.status_code} "
                f"duration={duration_ms:.2f}ms"
            )

        return response


class RequestIDFilter(logging.Filter):
    """Logging filter to add request ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Add request_id to log record if available
        record.request_id = request_id_context.get("")
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", ""),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code
        if hasattr(record, "method"):
            log_data["method"] = record.method
        if hasattr(record, "path"):
            log_data["path"] = record.path

        return json.dumps(log_data)


def setup_logging(log_level: str, log_format: str = "text"):
    """Configure logging with request ID support.

    Args:
        log_level: Logging level (debug, info, warning, error)
        log_format: "text" for human-readable, "json" for structured
    """
    # Create formatter based on format preference
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        # Text format with request ID
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler with formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIDFilter())
    root_logger.addHandler(console_handler)

    # Configure uvicorn loggers to use same format
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.addFilter(RequestIDFilter())
        logger.addHandler(handler)
        logger.propagate = False


def log_request_details(logger: logging.Logger, request_data: dict, request_id: str = ""):
    """Log request input details.

    Args:
        logger: Logger instance
        request_data: Request payload as dictionary
        request_id: Request ID for correlation
    """
    # Sanitize sensitive data (if any)
    sanitized = request_data.copy()

    # Log at INFO level with request ID
    logger.info(
        f"Request input: {json.dumps(sanitized, indent=2)}", extra={"request_id": request_id}
    )


def log_response_details(logger: logging.Logger, response_data: dict, request_id: str = ""):
    """Log response output details.

    Args:
        logger: Logger instance
        response_data: Response payload as dictionary
        request_id: Request ID for correlation
    """
    logger.info(
        f"Response output: {json.dumps(response_data, indent=2)}", extra={"request_id": request_id}
    )
