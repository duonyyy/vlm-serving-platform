"""Unit tests for Pydantic contracts and schemas."""

import json
import unittest
from packages.contracts.chat import (
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
from packages.contracts.errors import ErrorDetail, ErrorResponse
from packages.contracts.extraction import ExtractionRequest, ExtractionResponse
from packages.contracts.models import HealthResponse, ModelCapabilities, ModelCard, ModelListResponse


class ContractsTests(unittest.TestCase):
    def test_chat_completion_request_extraction(self) -> None:
        parts: list[ContentPart] = [
            TextContentPart(text="Đọc hóa đơn"),
            ImageContentPart(image_url=ImageUrl(url="data:image/png;base64,AABB")),
        ]
        msg = ChatMessage(role="user", content=parts)
        req = ChatCompletionRequest(
            model="qwen3-vl-2b",
            messages=[msg],
            max_tokens=128,
            temperature=0.0,
        )

        self.assertEqual(req.get_user_prompt(), "Đọc hóa đơn")
        self.assertEqual(req.get_image_urls(), ["data:image/png;base64,AABB"])

    def test_chat_completion_request_text_only(self) -> None:
        msg = ChatMessage(role="user", content="Xin chào")
        req = ChatCompletionRequest(model="qwen3-vl-2b", messages=[msg])
        self.assertEqual(req.get_user_prompt(), "Xin chào")
        self.assertEqual(req.get_image_urls(), [])

    def test_chat_completion_response_serialization(self) -> None:
        resp = ChatCompletionResponse(
            id="chatcmpl-test1234",
            model="qwen3-vl-2b",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatResponseMessage(content="Kết quả OCR"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            timing=TimingBreakdown(preprocessing_ms=5.0, inference_ms=45.0, total_ms=50.0),
        )
        data = resp.model_dump()
        self.assertEqual(data["id"], "chatcmpl-test1234")
        self.assertEqual(data["choices"][0]["message"]["content"], "Kết quả OCR")
        self.assertEqual(data["usage"]["total_tokens"], 15)

    def test_extraction_contracts(self) -> None:
        req = ExtractionRequest(
            model="qwen3-vl-2b",
            image_url="data:image/jpeg;base64,1234",
            schema_name="invoice",
            fields=["total_amount", "seller_name"],
        )
        self.assertEqual(req.schema_name, "invoice")
        self.assertEqual(req.fields, ["total_amount", "seller_name"])

        resp = ExtractionResponse(
            request_id="req_001",
            model="qwen3-vl-2b",
            status="completed",
            data={"total_amount": 50000, "seller_name": "ABC Store"},
        )
        self.assertEqual(resp.data["total_amount"], 50000)

    def test_model_and_health_contracts(self) -> None:
        card = ModelCard(id="qwen3-vl-2b", capabilities=ModelCapabilities(image_chat=True, ocr_markdown=True))
        models_resp = ModelListResponse(data=[card])
        self.assertEqual(len(models_resp.data), 1)
        self.assertTrue(models_resp.data[0].capabilities.ocr_markdown)

        health = HealthResponse(
            status="healthy",
            worker_id="worker_01",
            uptime_seconds=12.5,
            active_model="qwen3-vl-2b",
            active_model_status="READY",
            cuda_available=True,
            gpu={"device_name": "Tesla T4", "vram_allocated_bytes": 1000},
            timestamp="2026-10-26T22:00:00Z",
        )
        self.assertEqual(health.status, "healthy")
        self.assertEqual(health.gpu["device_name"], "Tesla T4")

    def test_error_response_contract(self) -> None:
        err = ErrorResponse(
            error=ErrorDetail(message="Image corrupted", type="invalid_request_error", code="bad_image"),
            request_id="req_err99",
        )
        data = err.model_dump()
        self.assertEqual(data["error"]["code"], "bad_image")
        self.assertEqual(data["request_id"], "req_err99")


if __name__ == "__main__":
    unittest.main()
