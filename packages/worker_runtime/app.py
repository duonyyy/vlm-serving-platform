"""FastAPI application for VLM Worker serving OpenAI-compatible API."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from packages.contracts.chat import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatResponseMessage,
    TimingBreakdown,
    UsageInfo,
)
from packages.contracts.errors import ErrorDetail, ErrorResponse
from packages.contracts.extraction import ExtractionRequest, ExtractionResponse
from packages.contracts.models import HealthResponse, ModelCard, ModelListResponse
from packages.image_processing.decoder import decode_image
from packages.telemetry.context import get_request_id, set_request_id
from packages.telemetry.logging import get_logger
from .manager import ModelManager

logger = get_logger("worker_api")


def create_app(manager: ModelManager | None = None) -> FastAPI:
    """Factory creating and configuring the FastAPI worker application."""
    model_manager = manager or ModelManager()

    app = FastAPI(
        title="VLM Worker Runtime API",
        version="0.1.0",
        description="Low-latency OpenAI-compatible serving engine for Vision-Language Models",
    )

    app.state.manager = model_manager

    # 1. Request Tracing & Timing Middleware
    @app.middleware("http")
    async def request_tracing_middleware(request: Request, call_next: Any) -> Response:
        start_time = time.perf_counter()
        req_id_header = request.headers.get("X-Request-ID")
        req_id = set_request_id(req_id_header)

        try:
            response = await call_next(request)
        except Exception as exc:
            # Let custom exception handlers format the error response
            raise exc

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = str(duration_ms)

        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)",
            extra={"extra_data": {"method": request.method, "path": request.url.path, "status_code": response.status_code, "duration_ms": duration_ms}},
        )
        return response

    # 2. Structured Error Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        req_id = get_request_id()
        error_msg = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors())
        logger.warning(f"Validation error: {error_msg}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error=ErrorDetail(message=error_msg, type="validation_error", code="invalid_payload"),
                request_id=req_id,
            ).model_dump(),
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        req_id = get_request_id()
        logger.warning(f"Bad request value error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error=ErrorDetail(message=str(exc), type="invalid_request_error", code="bad_input"),
                request_id=req_id,
            ).model_dump(),
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        req_id = get_request_id()
        err_str = str(exc)
        is_unavailable = any(k in err_str.lower() for k in ("not ready", "not registered", "no model adapter", "model unavailable"))
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE if is_unavailable else status.HTTP_500_INTERNAL_SERVER_ERROR
        logger.error(f"Runtime error ({status_code}): {exc}")
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=ErrorDetail(message=err_str, type="runtime_error", code="model_unavailable" if is_unavailable else "internal_failure"),
                request_id=req_id,
            ).model_dump(),
            headers={"X-Request-ID": req_id},
        )

    # 3. Routes
    @app.get("/health", response_model=HealthResponse)
    async def health_endpoint() -> HealthResponse:
        return model_manager.get_health()

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models_endpoint() -> ModelListResponse:
        cards = model_manager.list_models()
        return ModelListResponse(data=cards)

    @app.get("/v1/models/{model_id}", response_model=ModelCard)
    async def get_model_endpoint(model_id: str) -> ModelCard:
        cards = model_manager.list_models()
        for card in cards:
            if card.id == model_id:
                return card
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found on this worker",
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions_endpoint(payload: ChatCompletionRequest) -> ChatCompletionResponse:
        start_e2e = time.perf_counter()
        req_id = get_request_id()

        adapter = model_manager.get_adapter()
        user_prompt = payload.get_user_prompt()
        if not user_prompt:
            raise ValueError("Chat messages must contain at least one non-empty user prompt")

        # Decode image if provided
        image_urls = payload.get_image_urls()
        decoded_image = None
        prep_start = time.perf_counter()
        if image_urls:
            first_image_src = image_urls[0]
            try:
                decoded_image, _ = decode_image(first_image_src)
            except Exception as exc:
                raise ValueError(f"Failed to process input image: {exc}") from exc
        image_prep_time_ms = (time.perf_counter() - prep_start) * 1000

        # Concurrency-safe inference
        async with model_manager.lock:
            result = adapter.infer(
                prompt=user_prompt,
                image=decoded_image,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                **payload.extra_params,
            )

        total_elapsed_ms = round((time.perf_counter() - start_e2e) * 1000, 3)

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=adapter.model_id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatResponseMessage(content=result.prediction_text),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=result.input_tokens,
                completion_tokens=result.output_tokens,
                total_tokens=result.input_tokens + result.output_tokens,
            ),
            timing=TimingBreakdown(
                preprocessing_ms=round(result.preprocessing_ms + image_prep_time_ms, 3),
                inference_ms=round(result.inference_ms, 3),
                total_ms=total_elapsed_ms,
            ),
        )

    @app.post("/v1/extract", response_model=ExtractionResponse)
    async def structured_extract_endpoint(payload: ExtractionRequest) -> ExtractionResponse:
        start_e2e = time.perf_counter()
        req_id = get_request_id()

        adapter = model_manager.get_adapter()

        # Decode image
        decoded_image, _ = decode_image(payload.image_url)

        prompt = payload.prompt or (
            "Trích xuất thông tin hóa đơn sang JSON với các trường: seller_name, invoice_number, invoice_date, total_amount, currency."
            if payload.schema_name == "invoice"
            else f"Trích xuất các trường {payload.fields} sang JSON."
        )

        async with model_manager.lock:
            result = adapter.infer(prompt=prompt, image=decoded_image, max_tokens=512)

        # Attempt JSON parse of prediction
        parsed_data: dict[str, Any] = {}
        parse_status = "completed"
        try:
            # Look for JSON markdown block or raw JSON
            raw_text = result.prediction_text.strip()
            if "```json" in raw_text:
                json_part = raw_text.split("```json")[1].split("```")[0].strip()
                parsed_data = json.loads(json_part)
            elif "```" in raw_text:
                json_part = raw_text.split("```")[1].split("```")[0].strip()
                parsed_data = json.loads(json_part)
            elif raw_text.startswith("{") and raw_text.endswith("}"):
                parsed_data = json.loads(raw_text)
            else:
                parsed_data = {"raw": raw_text}
                parse_status = "partial"
        except Exception:
            parsed_data = {"raw": result.prediction_text}
            parse_status = "partial"

        total_elapsed_ms = round((time.perf_counter() - start_e2e) * 1000, 3)

        return ExtractionResponse(
            request_id=req_id,
            model=adapter.model_id,
            status=parse_status,
            data=parsed_data,
            raw_text=result.prediction_text,
            usage=UsageInfo(
                prompt_tokens=result.input_tokens,
                completion_tokens=result.output_tokens,
                total_tokens=result.input_tokens + result.output_tokens,
            ),
            timing=TimingBreakdown(
                preprocessing_ms=result.preprocessing_ms,
                inference_ms=result.inference_ms,
                total_ms=total_elapsed_ms,
            ),
        )

    return app
