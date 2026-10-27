"""Model metadata and worker health contracts."""

from __future__ import annotations

import time
from typing import Any, Literal
from pydantic import BaseModel, Field


class ModelCapabilities(BaseModel):
    image_chat: bool = True
    ocr_markdown: bool = False
    structured_json: bool = False
    max_image_edge: int = 1280
    supports_concurrency: bool = False


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vlm-platform"
    status: Literal["READY", "LOADING", "NOT_LOADED", "ERROR"] = "READY"
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    revision: str | None = None
    device: str | None = None


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    worker_id: str
    uptime_seconds: float
    active_model: str | None = None
    active_model_status: str = "NOT_LOADED"
    cuda_available: bool = False
    gpu: dict[str, Any] | None = None
    timestamp: str
