"""Data contracts and schemas for VLM serving."""

from .chat import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatResponseMessage,
    ContentPart,
    ImageContentPart,
    ImageUrl,
    TextContentPart,
    TimingBreakdown,
    UsageInfo,
)
from .errors import ErrorDetail, ErrorResponse
from .extraction import ExtractionRequest, ExtractionResponse, InvoiceData
from .models import HealthResponse, ModelCapabilities, ModelCard, ModelListResponse
from .protocol import (
    GatewayHeartbeatAck,
    GatewayJobCancel,
    GatewayJobDispatch,
    GatewayRegisterAck,
    ProtocolMessage,
    WorkerHeartbeat,
    WorkerJobResult,
    WorkerRegister,
)

__all__ = [
    "ImageUrl",
    "TextContentPart",
    "ImageContentPart",
    "ContentPart",
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatChoice",
    "ChatResponseMessage",
    "UsageInfo",
    "TimingBreakdown",
    "ChatCompletionResponse",
    "ExtractionRequest",
    "ExtractionResponse",
    "InvoiceData",
    "ModelCapabilities",
    "ModelCard",
    "ModelListResponse",
    "HealthResponse",
    "ErrorDetail",
    "ErrorResponse",
    "WorkerRegister",
    "GatewayRegisterAck",
    "WorkerHeartbeat",
    "GatewayHeartbeatAck",
    "GatewayJobDispatch",
    "WorkerJobResult",
    "GatewayJobCancel",
    "ProtocolMessage",
]
