"""Worker state, model lifecycle, and concurrency management."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from packages.contracts.models import HealthResponse, ModelCard
from packages.model_adapters.base import BaseModelAdapter


class ModelManager:
    """Manages adapter lifecycle, health reporting, and GPU concurrency locking."""

    def __init__(self, worker_id: str | None = None) -> None:
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.start_time = time.time()
        self.active_adapter: BaseModelAdapter | None = None
        self._lock = asyncio.Lock()

    @property
    def lock(self) -> asyncio.Lock:
        """Lock to enforce single-concurrency execution on GPU workers."""
        return self._lock

    def set_adapter(self, adapter: BaseModelAdapter, config: dict[str, Any] | None = None) -> None:
        """Register and load a model adapter."""
        if self.active_adapter is not None:
            self.active_adapter.unload()
        self.active_adapter = adapter
        self.active_adapter.load(config)

    def get_adapter(self) -> BaseModelAdapter:
        """Retrieve the active adapter, raising an error if uninitialized or not ready."""
        if self.active_adapter is None:
            raise RuntimeError("No model adapter is registered on this worker")
        if self.active_adapter.status != "READY":
            raise RuntimeError(f"Active model '{self.active_adapter.model_id}' is not READY (status: {self.active_adapter.status})")
        return self.active_adapter

    def get_health(self) -> HealthResponse:
        """Generate structured health report for liveness and readiness probes."""
        uptime = round(time.time() - self.start_time, 2)
        cuda_avail = False
        gpu_info = None

        try:
            import torch

            cuda_avail = torch.cuda.is_available()
            if cuda_avail and torch.cuda.device_count() > 0:
                gpu_info = {
                    "device_name": torch.cuda.get_device_name(0),
                    "vram_allocated_bytes": int(torch.cuda.memory_allocated(0)),
                    "vram_reserved_bytes": int(torch.cuda.memory_reserved(0)),
                }
        except (ImportError, Exception):
            pass

        if self.active_adapter is None:
            status = "degraded"
            model_id = None
            model_status = "NOT_LOADED"
        elif self.active_adapter.status == "READY":
            status = "healthy"
            model_id = self.active_adapter.model_id
            model_status = "READY"
        else:
            status = "unhealthy"
            model_id = self.active_adapter.model_id
            model_status = self.active_adapter.status

        return HealthResponse(
            status=status,
            worker_id=self.worker_id,
            uptime_seconds=uptime,
            active_model=model_id,
            active_model_status=model_status,
            cuda_available=cuda_avail,
            gpu=gpu_info,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def list_models(self) -> list[ModelCard]:
        """List active model information conforming to OpenAI /v1/models format."""
        if self.active_adapter is None:
            return []
        return [
            ModelCard(
                id=self.active_adapter.model_id,
                status=self.active_adapter.status,
                capabilities=self.active_adapter.capabilities,
            )
        ]
