"""Configuration settings for the VLM Gateway (Control Plane)."""

from __future__ import annotations

import os
from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    """Gateway runtime parameters loaded from environment or defaults."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    worker_auth_token: str = Field(
        default=os.getenv("VLM_WORKER_AUTH_TOKEN", "vlm_worker_secret_2026"),
        description="Shared secret required for workers to register",
    )
    client_api_key: str | None = Field(
        default=os.getenv("VLM_CLIENT_API_KEY", None),
        description="Optional API key required from clients",
    )
    queue_max_size: int = Field(default=100, description="Maximum concurrent pending jobs in queue")
    job_timeout_sec: float = Field(default=60.0, description="Maximum wait time for worker inference")
    heartbeat_timeout_sec: float = Field(default=20.0, description="TTL before worker is declared offline")
    enable_cache: bool = Field(default=True, description="Enable response caching for identical requests")
    cache_max_entries: int = Field(default=1000, description="Max entries stored in response cache")
    cache_ttl_sec: float = Field(default=3600.0, description="TTL for cached responses in seconds")
    scheduling_strategy: str = Field(default="least_loaded", description="Worker selection policy: least_loaded, round_robin, model_affinity")



default_config = GatewayConfig()
