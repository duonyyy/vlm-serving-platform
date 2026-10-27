"""Structured error schemas conforming to standard API conventions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    message: str = Field(description="Human-readable explanation of the error")
    type: str = Field(description="Error classification, e.g. 'invalid_request_error'")
    param: str | None = Field(default=None, description="The request parameter that caused the issue")
    code: str | None = Field(default=None, description="Specific error code string")


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str = Field(description="Unique request tracing ID")
