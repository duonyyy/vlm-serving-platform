"""OpenAI-compatible Multimodal Chat Completion schemas."""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field


class ImageUrl(BaseModel):
    """Image URL payload supporting data URI base64 or safe web URL."""

    url: str = Field(description="Direct URL or data URI containing base64 encoded image")
    detail: Literal["auto", "low", "high"] = Field(default="auto", description="Fidelity level")


class TextContentPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContentPart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = Annotated[
    Union[TextContentPart, ImageContentPart],
    Field(discriminator="type"),
]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class ChatCompletionRequest(BaseModel):
    model: str = Field(description="Model identifier, e.g. 'qwen3-vl-2b'")
    messages: list[ChatMessage] = Field(min_length=1, description="List of chat messages")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = Field(default=False)
    extra_params: dict[str, Any] = Field(default_factory=dict)

    def get_user_prompt(self) -> str:
        """Extract the text prompt from the user message(s)."""
        texts: list[str] = []
        for msg in self.messages:
            if msg.role in {"user", "system"}:
                if isinstance(msg.content, str):
                    texts.append(msg.content)
                elif isinstance(msg.content, list):
                    for part in msg.content:
                        if isinstance(part, TextContentPart):
                            texts.append(part.text)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part.get("text", ""))
        return " ".join(t.strip() for t in texts if t.strip())

    def get_image_urls(self) -> list[str]:
        """Extract all image URLs or base64 data URIs from messages."""
        urls: list[str] = []
        for msg in self.messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, ImageContentPart):
                        urls.append(part.image_url.url)
                    elif isinstance(part, dict) and part.get("type") == "image_url":
                        img_info = part.get("image_url", {})
                        if isinstance(img_info, dict) and "url" in img_info:
                            urls.append(img_info["url"])
                        elif isinstance(img_info, str):
                            urls.append(img_info)
        return urls


class ChatResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatResponseMessage
    finish_reason: Literal["stop", "length", "content_filter", "error"] = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TimingBreakdown(BaseModel):
    preprocessing_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0


class ChatCompletionResponse(BaseModel):
    id: str = Field(description="Unique completion ID, e.g. 'chatcmpl-...'")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo
    timing: TimingBreakdown | None = None
    system_fingerprint: str | None = None
