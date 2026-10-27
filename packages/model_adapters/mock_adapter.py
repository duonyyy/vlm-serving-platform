"""High-fidelity Mock Adapter for testing, CI, and local CPU verification."""

from __future__ import annotations

import json
import time
from typing import Any
from PIL import Image

from .base import BaseModelAdapter, InferenceResult


class MockAdapter(BaseModelAdapter):
    """Mock VLM Adapter that simulates realistic model responses and timings."""

    def __init__(self, model_id: str = "mock-vlm-2b") -> None:
        super().__init__(model_id)
        self.simulate_latency_ms: float = 10.0
        self.simulate_oom: bool = False
        self.fail_on_prompt: str | None = None
        self._capabilities.image_chat = True
        self._capabilities.ocr_markdown = True
        self._capabilities.structured_json = True

    def load(self, config: dict[str, Any] | None = None) -> None:
        self._status = "LOADING"
        cfg = config or {}
        if cfg.get("fail_on_load"):
            self._status = "ERROR"
            self._error_message = "Simulated weight loading failure"
            raise RuntimeError(self._error_message)

        self.simulate_latency_ms = float(cfg.get("simulate_latency_ms", 10.0))
        self.simulate_oom = bool(cfg.get("simulate_oom", False))
        self.fail_on_prompt = cfg.get("fail_on_prompt")
        self._status = "READY"
        self._error_message = None

    def infer(
        self,
        prompt: str,
        image: Image.Image | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> InferenceResult:
        if self._status != "READY":
            raise RuntimeError(f"Model {self.model_id} is not in READY state (current: {self._status})")

        if self.simulate_oom:
            raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB (GPU 0; 15.78 GiB total capacity)")

        if self.fail_on_prompt and self.fail_on_prompt in prompt:
            raise ValueError(f"Intentional failure triggered by prompt keyword: {self.fail_on_prompt}")

        start = time.perf_counter()
        prep_start = time.perf_counter()

        image_size = image.size if image else (0, 0)
        # Simulate slight prep time
        time.sleep(0.002)
        prep_ms = (time.perf_counter() - prep_start) * 1000

        infer_start = time.perf_counter()
        # Simulate inference time
        if self.simulate_latency_ms > 0:
            time.sleep(self.simulate_latency_ms / 1000.0)

        # Generate realistic output based on prompt context
        prompt_lower = prompt.lower()
        if "hóa đơn" in prompt_lower or "invoice" in prompt_lower or "json" in prompt_lower:
            prediction = json.dumps(
                {
                    "seller_name": "Công ty TNHH Cà Phê Việt",
                    "invoice_number": "HD-2026-9921",
                    "invoice_date": "2026-10-13",
                    "total_amount": 150000.0,
                    "currency": "VND",
                },
                ensure_ascii=False,
            )
        elif "mô tả" in prompt_lower or "describe" in prompt_lower:
            prediction = f"Ảnh có kích thước {image_size[0]}x{image_size[1]}, hiển thị cảnh quan với ánh sáng tự nhiên rõ nét."
        elif "ocr" in prompt_lower:
            prediction = "# BÁO CÁO TÀI CHÍNH\nNăm tài khóa 2026\nDoanh thu: 1.250.000.000 VND"
        else:
            prediction = f"Phản hồi từ {self.model_id}: Tôi đã nhận được câu hỏi '{prompt}' và xử lý ảnh {image_size} thành công."

        infer_ms = (time.perf_counter() - infer_start) * 1000

        # Estimate token counts
        input_tokens = len(prompt.split()) + (256 if image else 0)
        output_tokens = len(prediction.split())

        return InferenceResult(
            prediction_text=prediction,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            preprocessing_ms=round(prep_ms, 3),
            inference_ms=round(infer_ms, 3),
            finish_reason="stop",
            extra={
                "mock": True,
                "image_width": image_size[0],
                "image_height": image_size[1],
            },
        )

    def unload(self) -> None:
        self._status = "NOT_LOADED"
        self._error_message = None
