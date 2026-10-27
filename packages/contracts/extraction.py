"""Structured Document & Invoice Extraction schemas."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

from .chat import TimingBreakdown, UsageInfo


class ExtractionRequest(BaseModel):
    model: str = Field(description="Model identifier, e.g. 'qwen3-vl-2b' or 'paddleocr-vl-1.6'")
    image_url: str = Field(description="Image base64 data URI or image URL")
    prompt: str | None = Field(default=None, description="Custom prompt or extraction instructions")
    schema_name: str | None = Field(default="invoice", description="Target extraction schema")
    fields: list[str] | None = Field(default=None, description="Optional specific fields to extract")


class InvoiceData(BaseModel):
    seller_name: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    subtotal: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None
    currency: str = "VND"
    items: list[dict[str, Any]] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    request_id: str
    model: str
    status: Literal["completed", "failed", "partial"] = "completed"
    data: dict[str, Any] = Field(default_factory=dict)
    raw_text: str | None = None
    usage: UsageInfo = Field(default_factory=UsageInfo)
    timing: TimingBreakdown | None = None
