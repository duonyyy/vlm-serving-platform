"""Python Client SDK for communicating with VLM Serving Worker / Gateway."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any
import httpx
from PIL import Image

from packages.contracts.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ContentPart,
    ImageContentPart,
    ImageUrl,
    TextContentPart,
)
from packages.contracts.extraction import ExtractionRequest, ExtractionResponse
from packages.contracts.models import HealthResponse, ModelCard, ModelListResponse


class VLMClientError(Exception):
    """Base exception for VLM client errors."""


class VLMConnectionError(VLMClientError):
    """Raised when connection to server fails."""


class VLMAPIError(VLMClientError):
    """Raised when API returns a 4xx or 5xx status error."""

    def __init__(self, message: str, status_code: int, request_id: str | None = None, error_type: str | None = None) -> None:
        super().__init__(f"[{status_code}] (req: {request_id}): {message}")
        self.status_code = status_code
        self.request_id = request_id
        self.error_type = error_type


class VLMValidationError(VLMClientError):
    """Raised on local request validation or image formatting failure."""


def image_to_data_uri(image_input: str | bytes | Path | Image.Image) -> str:
    """Convert an image (file path, raw bytes, Pillow Image, or existing data URI) into a base64 data URI."""
    if isinstance(image_input, str):
        if image_input.startswith("data:") or image_input.startswith(("http://", "https://")):
            return image_input
        path = Path(image_input)
        if path.is_file():
            raw_bytes = path.read_bytes()
            suffix = path.suffix.lower().replace(".", "")
            fmt = "jpeg" if suffix in {"jpg", "jpeg"} else ("png" if suffix == "png" else "webp")
            b64 = base64.b64encode(raw_bytes).decode("ascii")
            return f"data:image/{fmt};base64,{b64}"
        else:
            raise VLMValidationError(f"Image path does not exist: {path}")

    elif isinstance(image_input, Path):
        if not image_input.is_file():
            raise VLMValidationError(f"Image file does not exist: {image_input}")
        raw_bytes = image_input.read_bytes()
        suffix = image_input.suffix.lower().replace(".", "")
        fmt = "jpeg" if suffix in {"jpg", "jpeg"} else ("png" if suffix == "png" else "webp")
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        return f"data:image/{fmt};base64,{b64}"

    elif isinstance(image_input, bytes):
        b64 = base64.b64encode(image_input).decode("ascii")
        return f"data:image/png;base64,{b64}"

    elif isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    raise VLMValidationError(f"Unsupported image type: {type(image_input).__name__}")


class VLMClient:
    """Synchronous Python Client for VLM Serving API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 60.0,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self._custom_client = http_client
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"

    def _request(
        self,
        method: str,
        path: str,
        json_data: Any = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        headers = dict(self._headers)
        if extra_headers:
            headers.update(extra_headers)

        try:
            if self._custom_client is not None:
                resp = self._custom_client.request(method, path, json=json_data, headers=headers)
            else:
                with httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout) as client:
                    resp = client.request(method, path, json=json_data)
        except Exception as exc:
            raise VLMConnectionError(f"Failed to connect to {self.base_url}: {exc}") from exc

        if resp.status_code >= 400:
            self._handle_error_response(resp)
        return resp

    def get_health(self) -> HealthResponse:
        """Check worker health and status."""
        resp = self._request("GET", "/health")
        return HealthResponse.model_validate(resp.json())

    def list_models(self) -> list[ModelCard]:
        """List available models on this worker/gateway."""
        resp = self._request("GET", "/v1/models")
        return ModelListResponse.model_validate(resp.json()).data

    def chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        model: str = "qwen3-vl-2b",
        max_tokens: int = 256,
        temperature: float = 0.0,
        request_id: str | None = None,
        **extra: Any,
    ) -> ChatCompletionResponse:
        """Send multimodal chat completion request."""
        parsed_msgs: list[ChatMessage] = []
        for m in messages:
            if isinstance(m, ChatMessage):
                parsed_msgs.append(m)
            elif isinstance(m, dict):
                parsed_msgs.append(ChatMessage.model_validate(m))
            else:
                raise VLMValidationError(f"Invalid message format: {m}")

        req_payload = ChatCompletionRequest(
            model=model,
            messages=parsed_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_params=extra,
        )

        headers: dict[str, str] = {}
        if request_id:
            headers["X-Request-ID"] = request_id

        resp = self._request("POST", "/v1/chat/completions", json_data=req_payload.model_dump(), extra_headers=headers)
        return ChatCompletionResponse.model_validate(resp.json())

    def chat_image(
        self,
        image: str | bytes | Path | Image.Image,
        prompt: str,
        model: str = "qwen3-vl-2b",
        max_tokens: int = 256,
        temperature: float = 0.0,
        request_id: str | None = None,
        **extra: Any,
    ) -> ChatCompletionResponse:
        """Convenience method to query a single image with text prompt."""
        data_uri = image_to_data_uri(image)
        content_parts: list[ContentPart] = [
            TextContentPart(text=prompt),
            ImageContentPart(image_url=ImageUrl(url=data_uri)),
        ]
        msg = ChatMessage(role="user", content=content_parts)
        return self.chat(
            messages=[msg],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            request_id=request_id,
            **extra,
        )

    def extract(
        self,
        image: str | bytes | Path | Image.Image,
        prompt: str | None = None,
        schema_name: str = "invoice",
        fields: list[str] | None = None,
        model: str = "qwen3-vl-2b",
        request_id: str | None = None,
    ) -> ExtractionResponse:
        """Convenience method to extract structured data from an image."""
        data_uri = image_to_data_uri(image)
        req_payload = ExtractionRequest(
            model=model,
            image_url=data_uri,
            prompt=prompt,
            schema_name=schema_name,
            fields=fields,
        )
        headers: dict[str, str] = {}
        if request_id:
            headers["X-Request-ID"] = request_id

        resp = self._request("POST", "/v1/extract", json_data=req_payload.model_dump(), extra_headers=headers)
        return ExtractionResponse.model_validate(resp.json())

    def _handle_error_response(self, resp: httpx.Response) -> None:
        req_id = resp.headers.get("X-Request-ID")
        try:
            body = resp.json()
            err_dict = body.get("error", {})
            msg = err_dict.get("message", resp.text)
            err_type = err_dict.get("type", "api_error")
            req_id = body.get("request_id", req_id)
        except Exception:
            msg = resp.text
            err_type = "api_error"
        raise VLMAPIError(message=msg, status_code=resp.status_code, request_id=req_id, error_type=err_type)
