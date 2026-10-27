"""Context propagation for request tracing across async operations."""

from __future__ import annotations

import contextvars
import uuid

_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def get_request_id() -> str:
    """Retrieve the current request ID from context, or empty string if unset."""
    return _request_id_ctx.get()


def set_request_id(request_id: str | None = None) -> str:
    """Set the request ID in the context. Generates a new unique ID if not provided."""
    req_id = request_id.strip() if request_id and request_id.strip() else f"req_{uuid.uuid4().hex[:12]}"
    _request_id_ctx.set(req_id)
    return req_id
