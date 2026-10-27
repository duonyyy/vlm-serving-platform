"""WebSocket protocol contracts between Gateway (Control Plane) and Workers."""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field

from .chat import TimingBreakdown
from .models import ModelCapabilities


class WorkerRegister(BaseModel):
    """Initial handshake message sent by the worker upon connecting to the Gateway."""

    type: Literal["register"] = "register"
    worker_id: str
    model_id: str
    token: str = Field(description="Shared secret token for worker authentication")
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    device_info: dict[str, Any] = Field(default_factory=dict)


class GatewayRegisterAck(BaseModel):
    """Registration response returned by Gateway to Worker."""

    type: Literal["register_ack"] = "register_ack"
    status: Literal["registered", "rejected"]
    session_id: str = Field(default="")
    heartbeat_interval_sec: float = 5.0
    reason: str | None = None


class WorkerHeartbeat(BaseModel):
    """Periodic health ping sent by Worker to Gateway."""

    type: Literal["heartbeat"] = "heartbeat"
    worker_id: str
    session_id: str
    timestamp: float = Field(default_factory=time.time)
    status: Literal["idle", "busy", "unhealthy"] = "idle"
    gpu_vram_allocated_bytes: int | None = None


class GatewayHeartbeatAck(BaseModel):
    """Heartbeat response from Gateway."""

    type: Literal["heartbeat_ack"] = "heartbeat_ack"
    timestamp: float = Field(default_factory=time.time)


class GatewayJobDispatch(BaseModel):
    """Job assignment sent by Gateway to an idle Worker."""

    type: Literal["job_dispatch"] = "job_dispatch"
    job_id: str
    request_id: str
    endpoint: str = Field(description="/v1/chat/completions or /v1/extract")
    payload: dict[str, Any]
    timeout_sec: float = 60.0


class WorkerJobResult(BaseModel):
    """Inference execution result returned by Worker to Gateway."""

    type: Literal["job_result"] = "job_result"
    job_id: str
    request_id: str
    status: Literal["completed", "failed"]
    result: dict[str, Any] | None = None
    error: str | None = None
    timing: TimingBreakdown | None = None


class GatewayJobCancel(BaseModel):
    """Cancellation notification from Gateway (e.g. client disconnect or timeout)."""

    type: Literal["job_cancel"] = "job_cancel"
    job_id: str
    reason: str


ProtocolMessage = Annotated[
    Union[
        WorkerRegister,
        GatewayRegisterAck,
        WorkerHeartbeat,
        GatewayHeartbeatAck,
        GatewayJobDispatch,
        WorkerJobResult,
        GatewayJobCancel,
    ],
    Field(discriminator="type"),
]
