"""FastAPI Gateway Application (Control Plane) for Remote VLM Workers."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from packages.contracts.chat import ChatCompletionRequest, ChatCompletionResponse
from packages.contracts.errors import ErrorDetail, ErrorResponse
from packages.contracts.extraction import ExtractionRequest, ExtractionResponse
from packages.contracts.models import ModelListResponse
from packages.contracts.protocol import (
    GatewayHeartbeatAck,
    GatewayRegisterAck,
    WorkerHeartbeat,
    WorkerJobResult,
    WorkerRegister,
)
from packages.caching.cache import InferenceCache
from packages.telemetry.context import get_request_id, set_request_id
from packages.telemetry.logging import get_logger
from packages.telemetry.metrics import global_metrics


from .config import GatewayConfig, default_config
from .dispatcher import (
    JobDispatcher,
    JobTimeoutError,
    NoWorkerAvailableError,
    QueueFullError,
    RemoteWorkerExecutionError,
)
from .registry import WorkerRegistry
from .scheduler import SchedulingStrategy, WorkerScheduler

logger = get_logger("gateway_app")


def create_gateway_app(config: GatewayConfig | None = None) -> FastAPI:
    """Factory creating and configuring the Gateway FastAPI application."""
    cfg = config or default_config

    registry = WorkerRegistry(
        auth_token=cfg.worker_auth_token,
        heartbeat_ttl_sec=cfg.heartbeat_timeout_sec,
    )
    cache = (
        InferenceCache(max_entries=cfg.cache_max_entries, default_ttl_sec=cfg.cache_ttl_sec)
        if cfg.enable_cache
        else None
    )
    scheduler = WorkerScheduler(strategy=SchedulingStrategy(cfg.scheduling_strategy))
    dispatcher = JobDispatcher(
        registry=registry,
        max_queue_size=cfg.queue_max_size,
        default_timeout_sec=cfg.job_timeout_sec,
        cache=cache,
        scheduler=scheduler,
    )

    app = FastAPI(

        title="VLM Serving Gateway",
        description="Control plane dispatching inference jobs across outbound-connected remote workers.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach instances to app state for testing and inspection
    app.state.config = cfg
    app.state.registry = registry
    app.state.dispatcher = dispatcher

    @app.middleware("http")
    async def request_id_and_timing_middleware(request: Request, call_next: Any) -> Response:
        req_id = request.headers.get("x-request-id") or f"gw_{uuid.uuid4().hex[:12]}"
        set_request_id(req_id)
        start_time = time.time()

        response = await call_next(request)
        duration_sec = time.time() - start_time
        duration_ms = duration_sec * 1000.0
        response.headers["x-request-id"] = req_id
        response.headers["x-gateway-latency-ms"] = f"{duration_ms:.2f}"

        # Record metrics
        global_metrics.record_request(
            endpoint=request.url.path,
            model="default",
            status_code=response.status_code,
            duration_sec=duration_sec,
        )
        global_metrics.set_active_workers(registry.active_worker_count)
        global_metrics.set_pending_jobs(dispatcher.pending_count)

        return response

    async def verify_client_api_key(authorization: str | None = Header(None)) -> None:
        """Validate optional Bearer token from external client callers."""
        if not cfg.client_api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail={"error": {"message": "Missing Bearer token", "type": "AuthenticationError"}},
            )
        token = authorization.split("Bearer ")[1].strip()
        if token != cfg.client_api_key:
            raise HTTPException(
                status_code=401,
                detail={"error": {"message": "Invalid API key", "type": "AuthenticationError"}},
            )

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        """Expose Prometheus format platform telemetry."""
        global_metrics.set_active_workers(registry.active_worker_count)
        global_metrics.set_pending_jobs(dispatcher.pending_count)
        content = global_metrics.render_prometheus_text()
        return Response(content=content, media_type="text/plain; version=0.0.4")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Gateway health check and worker cluster status."""

        active_workers = registry.active_worker_count
        return {
            "status": "healthy" if active_workers > 0 else "degraded",
            "active_workers": active_workers,
            "pending_jobs": dispatcher.pending_count,
            "timestamp": time.time(),
        }

    @app.get("/v1/models", dependencies=[Depends(verify_client_api_key)])
    async def list_models() -> ModelListResponse:
        """List active models served by currently connected remote workers."""
        models = registry.list_models()
        return ModelListResponse(data=models)

    @app.get("/v1/cache/stats", dependencies=[Depends(verify_client_api_key)])
    async def cache_stats() -> dict[str, Any]:
        """Return cache hit/miss statistics and memory footprint."""
        if not dispatcher.cache:
            return {"enabled": False}
        stats = dispatcher.cache.stats
        return {
            "enabled": True,
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
            "total_entries": stats.total_entries,
            "max_entries": stats.max_entries,
            "saved_latency_ms": stats.saved_latency_ms,
        }

    @app.post("/v1/cache/clear", dependencies=[Depends(verify_client_api_key)])
    async def clear_cache() -> dict[str, str]:
        """Manually flush the inference response cache."""
        if dispatcher.cache:
            dispatcher.cache.clear()
            return {"status": "cleared"}
        return {"status": "cache_disabled"}



    @app.post("/v1/chat/completions", dependencies=[Depends(verify_client_api_key)])
    async def chat_completions(req: ChatCompletionRequest) -> Any:
        """OpenAI-compatible chat completions endpoint routed to a remote worker."""
        request_id = get_request_id() or f"req_{uuid.uuid4().hex[:12]}"
        try:
            result = await dispatcher.dispatch_job(
                endpoint="/v1/chat/completions",
                payload=req.model_dump(exclude_none=True),
                request_id=request_id,
                model_id=req.model,
                timeout_sec=cfg.job_timeout_sec,
            )
            return JSONResponse(status_code=200, content=result)
        except NoWorkerAvailableError as exc:
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error=ErrorDetail(message=str(exc), type="ServiceUnavailableError", code="no_workers_available"),
                    request_id=request_id,
                ).model_dump(),
            )
        except QueueFullError as exc:
            return JSONResponse(
                status_code=429,
                content=ErrorResponse(
                    error=ErrorDetail(message=str(exc), type="RateLimitError", code="gateway_queue_full"),
                    request_id=request_id,
                ).model_dump(),
            )
        except JobTimeoutError as exc:
            return JSONResponse(
                status_code=504,
                content=ErrorResponse(
                    error=ErrorDetail(message=str(exc), type="TimeoutError", code="worker_timeout"),
                    request_id=request_id,
                ).model_dump(),
            )
        except RemoteWorkerExecutionError as exc:
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(message=str(exc), type="InternalServerError", code="remote_worker_error"),
                    request_id=request_id,
                ).model_dump(),
            )

    @app.post("/v1/extract", dependencies=[Depends(verify_client_api_key)])
    async def extract(req: ExtractionRequest) -> Any:
        """Document information extraction endpoint routed to a remote worker."""
        request_id = get_request_id() or f"req_{uuid.uuid4().hex[:12]}"
        try:
            result = await dispatcher.dispatch_job(
                endpoint="/v1/extract",
                payload=req.model_dump(exclude_none=True),
                request_id=request_id,
                model_id=req.model,
                timeout_sec=cfg.job_timeout_sec,
            )
            return JSONResponse(status_code=200, content=result)
        except NoWorkerAvailableError as exc:
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error=ErrorDetail(message=str(exc), type="ServiceUnavailableError", code="no_workers_available"),
                    request_id=request_id,
                ).model_dump(),
            )
        except QueueFullError as exc:
            return JSONResponse(
                status_code=429,
                content=ErrorResponse(
                    error=ErrorDetail(message=str(exc), type="RateLimitError", code="gateway_queue_full"),
                    request_id=request_id,
                ).model_dump(),
            )
        except JobTimeoutError as exc:
            return JSONResponse(
                status_code=504,
                content=ErrorResponse(
                    error=ErrorDetail(message=str(exc), type="TimeoutError", code="worker_timeout"),
                    request_id=request_id,
                ).model_dump(),
            )
        except RemoteWorkerExecutionError as exc:
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(message=str(exc), type="InternalServerError", code="remote_worker_error"),
                    request_id=request_id,
                ).model_dump(),
            )


    @app.websocket("/workers/connect")
    async def worker_websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint receiving outbound connections from remote workers."""
        await websocket.accept()
        session_id: str | None = None

        try:
            # Step 1: Handshake
            init_text = await websocket.receive_text()
            data = json.loads(init_text)
            if data.get("type") != "register":
                await websocket.send_text(
                    GatewayRegisterAck(
                        status="rejected", reason="First message must be type 'register'"
                    ).model_dump_json()
                )
                await websocket.close(code=4000)
                return

            reg_msg = WorkerRegister.model_validate(data)
            success, sid, reason = registry.register(reg_msg, websocket=websocket)

            if not success:
                await websocket.send_text(
                    GatewayRegisterAck(status="rejected", reason=reason).model_dump_json()
                )
                await websocket.close(code=4001)
                return

            session_id = sid
            await websocket.send_text(
                GatewayRegisterAck(
                    status="registered",
                    session_id=session_id,
                    heartbeat_interval_sec=5.0,
                ).model_dump_json()
            )

            # Step 2: Message loop
            while True:
                msg_text = await websocket.receive_text()
                msg_data = json.loads(msg_text)
                msg_type = msg_data.get("type")

                if msg_type == "heartbeat":
                    hb = WorkerHeartbeat.model_validate(msg_data)
                    registry.record_heartbeat(hb)
                    await websocket.send_text(GatewayHeartbeatAck().model_dump_json())

                elif msg_type == "job_result":
                    res = WorkerJobResult.model_validate(msg_data)
                    dispatcher.resolve_job(res)

                else:
                    logger.warning(f"Unrecognized message type '{msg_type}' from session '{session_id}'")

        except WebSocketDisconnect:
            logger.info(f"Worker session '{session_id}' disconnected")
        except Exception as exc:
            logger.error(f"Error in worker session '{session_id}': {exc}")
        finally:
            if session_id:
                dispatcher.handle_worker_disconnect(session_id)
                registry.deregister(session_id)

    return app


app = create_gateway_app()
