"""Worker Registry and Session Management for Gateway Control Plane."""

from __future__ import annotations

import time
import uuid
from typing import Any
from fastapi import WebSocket
from pydantic import BaseModel, ConfigDict, Field

from packages.contracts.models import ModelCapabilities, ModelCard
from packages.contracts.protocol import WorkerHeartbeat, WorkerRegister
from packages.telemetry.logging import get_logger

logger = get_logger("gateway_registry")


class ConnectedWorker(BaseModel):
    """Active remote worker session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    worker_id: str
    model_id: str
    capabilities: ModelCapabilities
    device_info: dict[str, Any] = Field(default_factory=dict)
    connected_at: float = Field(default_factory=time.time)
    last_heartbeat: float = Field(default_factory=time.time)
    status: str = "idle"  # idle, busy, unhealthy
    websocket: Any = None
    active_job_id: str | None = None


class WorkerRegistry:
    """Thread-safe registry for connected remote workers."""

    def __init__(self, auth_token: str, heartbeat_ttl_sec: float = 20.0) -> None:
        self.auth_token = auth_token
        self.heartbeat_ttl_sec = heartbeat_ttl_sec
        self.sessions: dict[str, ConnectedWorker] = {}

    def register(
        self,
        reg: WorkerRegister,
        websocket: WebSocket,
    ) -> tuple[bool, str, str | None]:
        """Authenticate and register a remote worker.

        Returns:
            (success, session_id, rejection_reason)
        """
        if reg.token != self.auth_token:
            logger.warning(f"Worker '{reg.worker_id}' rejected: invalid authentication token")
            return False, "", "Authentication failed: invalid worker_auth_token"

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        worker = ConnectedWorker(
            session_id=session_id,
            worker_id=reg.worker_id,
            model_id=reg.model_id,
            capabilities=reg.capabilities,
            device_info=reg.device_info,
            websocket=websocket,
        )
        self.sessions[session_id] = worker
        logger.info(f"Worker '{reg.worker_id}' registered successfully (session: {session_id}, model: {reg.model_id})")
        return True, session_id, None

    def record_heartbeat(self, hb: WorkerHeartbeat) -> bool:
        """Update last heartbeat timestamp for session."""
        worker = self.sessions.get(hb.session_id)
        if not worker:
            return False
        worker.last_heartbeat = time.time()
        worker.status = hb.status
        return True

    def deregister(self, session_id: str) -> None:
        """Remove a disconnected worker."""
        if session_id in self.sessions:
            w = self.sessions.pop(session_id)
            logger.info(f"Worker '{w.worker_id}' (session: {session_id}) deregistered")

    def get_available_worker(self, model_id: str | None = None) -> ConnectedWorker | None:
        """Find an idle, healthy worker matching the requested model."""
        self.prune_stale_workers()
        candidates = [
            w for w in self.sessions.values()
            if w.status == "idle" and (model_id is None or w.model_id == model_id or model_id in w.model_id)
        ]
        return candidates[0] if candidates else None

    def prune_stale_workers(self) -> list[str]:
        """Disconnect and remove workers whose heartbeat timed out."""
        now = time.time()
        stale_ids = [
            sess_id for sess_id, w in self.sessions.items()
            if (now - w.last_heartbeat) > self.heartbeat_ttl_sec
        ]
        for sid in stale_ids:
            logger.warning(f"Pruning stale worker session '{sid}' (heartbeat timed out)")
            self.deregister(sid)
        return stale_ids

    def list_models(self) -> list[ModelCard]:
        """Return distinct models served across all active workers."""
        self.prune_stale_workers()
        seen_models: dict[str, ModelCard] = {}
        for w in self.sessions.values():
            if w.model_id not in seen_models:
                seen_models[w.model_id] = ModelCard(
                    id=w.model_id,
                    status="READY",
                    capabilities=w.capabilities,
                    device=w.device_info.get("name"),
                )
        return list(seen_models.values())

    @property
    def active_worker_count(self) -> int:
        self.prune_stale_workers()
        return len(self.sessions)
