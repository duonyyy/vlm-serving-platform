"""Outbound WebSocket Remote Worker Client for Kaggle and Remote GPUs."""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from packages.contracts.chat import ChatCompletionRequest, ChatCompletionResponse
from packages.contracts.extraction import ExtractionRequest, ExtractionResponse
from packages.contracts.protocol import (
    GatewayJobDispatch,
    GatewayRegisterAck,
    WorkerHeartbeat,
    WorkerJobResult,
    WorkerRegister,
)
from packages.model_adapters.base import BaseModelAdapter
from packages.telemetry.logging import get_logger

logger = get_logger("remote_worker_client")


def get_gpu_info() -> dict[str, Any]:
    """Retrieve PyTorch CUDA device metadata if available."""
    try:
        import torch

        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev_idx)
            return {
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "cuda_available": True,
            }
    except Exception:
        pass
    return {"name": "CPU/Mock", "cuda_available": False}


class RemoteWorkerClient:
    """Client running on Kaggle or a remote GPU server that establishes an

    outbound WebSocket connection back to the Gateway Control Plane.
    """

    def __init__(
        self,
        gateway_ws_url: str,
        adapter: BaseModelAdapter,
        auth_token: str,
        worker_id: str | None = None,
        max_reconnect_delay_sec: float = 30.0,
    ) -> None:
        self.gateway_ws_url = gateway_ws_url
        self.adapter = adapter
        self.auth_token = auth_token
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.max_reconnect_delay_sec = max_reconnect_delay_sec

        self.session_id: str | None = None
        self._execution_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        """Signal client to terminate cleanly."""
        self._stop_event.set()

    async def run(self) -> None:
        """Main connection and auto-reconnect loop with exponential backoff."""
        reconnect_attempts = 0
        base_delay = 1.0

        logger.info(f"Starting RemoteWorkerClient '{self.worker_id}' -> {self.gateway_ws_url}")

        while not self._stop_event.is_set():
            try:
                await self._connect_and_serve()
                reconnect_attempts = 0  # Reset on successful run
            except (ConnectionClosed, OSError, Exception) as exc:
                if self._stop_event.is_set():
                    break

                reconnect_attempts += 1
                backoff = min(base_delay * (2 ** (reconnect_attempts - 1)), self.max_reconnect_delay_sec)
                jitter = random.uniform(0.1, 1.0)
                wait_time = backoff + jitter

                logger.warning(
                    f"Connection lost to Gateway ({exc}). Reconnecting in {wait_time:.1f}s (attempt {reconnect_attempts})..."
                )
                await asyncio.sleep(wait_time)

    async def _connect_and_serve(self) -> None:
        """Connect WebSocket, complete handshake, and run heartbeat + dispatch tasks."""
        async with websockets.connect(self.gateway_ws_url) as ws:
            # 1. Handshake
            reg_msg = WorkerRegister(
                worker_id=self.worker_id,
                model_id=self.adapter.model_id,
                token=self.auth_token,
                capabilities=self.adapter.get_capabilities(),
                device_info=get_gpu_info(),
            )
            await ws.send(reg_msg.model_dump_json())

            ack_data = json.loads(await ws.recv())
            ack = GatewayRegisterAck.model_validate(ack_data)

            if ack.status != "registered":
                logger.error(f"Gateway registration rejected: {ack.reason}")
                raise RuntimeError(f"Registration rejected: {ack.reason}")

            self.session_id = ack.session_id
            logger.info(f"Registered with session '{self.session_id}'. Heartbeat interval: {ack.heartbeat_interval_sec}s")

            # 2. Spawn concurrent heartbeat and message handling tasks
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws, ack.heartbeat_interval_sec))
            listener_task = asyncio.create_task(self._listener_loop(ws))

            done, pending = await asyncio.wait(
                [heartbeat_task, listener_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            for task in done:
                # Re-raise exceptions if any
                exc = task.exception()
                if exc:
                    raise exc

    async def _heartbeat_loop(self, ws: Any, interval_sec: float) -> None:
        """Periodically send health heartbeat pings to Gateway."""
        while not self._stop_event.is_set():
            await asyncio.sleep(interval_sec)
            status = "busy" if self._execution_lock.locked() else "idle"
            hb = WorkerHeartbeat(
                worker_id=self.worker_id,
                session_id=self.session_id or "",
                status=status,
            )
            await ws.send(hb.model_dump_json())

    async def _listener_loop(self, ws: Any) -> None:
        """Listen for job dispatch commands from Gateway and execute inference."""
        while not self._stop_event.is_set():
            msg_text = await ws.recv()
            data = json.loads(msg_text)
            msg_type = data.get("type")

            if msg_type == "heartbeat_ack":
                continue

            if msg_type == "job_dispatch":
                dispatch = GatewayJobDispatch.model_validate(data)
                # Execute inference under single-concurrency lock to protect GPU VRAM
                asyncio.create_task(self._execute_and_respond(ws, dispatch))

            elif msg_type == "job_cancel":
                logger.info(f"Received job cancellation: {data.get('job_id')}")

    async def _execute_and_respond(self, ws: Any, dispatch: GatewayJobDispatch) -> None:
        """Execute inference on the adapter and send WorkerJobResult back."""
        async with self._execution_lock:
            logger.info(f"Executing job '{dispatch.job_id}' for endpoint '{dispatch.endpoint}'")
            start_t = time.time()
            try:
                if dispatch.endpoint == "/v1/chat/completions":
                    req = ChatCompletionRequest.model_validate(dispatch.payload)
                    loop = asyncio.get_running_loop()
                    # Run sync adapter.infer in thread pool to avoid blocking the asyncio event loop
                    resp: ChatCompletionResponse = await loop.run_in_executor(
                        None, lambda: self.adapter.infer(req)
                    )
                    job_result = WorkerJobResult(
                        job_id=dispatch.job_id,
                        request_id=dispatch.request_id,
                        status="completed",
                        result=resp.model_dump(),
                        timing=resp.timing,
                    )
                elif dispatch.endpoint == "/v1/extract":
                    req_extract = ExtractionRequest.model_validate(dispatch.payload)
                    loop = asyncio.get_running_loop()
                    resp_extract: ExtractionResponse = await loop.run_in_executor(
                        None, lambda: self.adapter.extract(req_extract)
                    )
                    job_result = WorkerJobResult(
                        job_id=dispatch.job_id,
                        request_id=dispatch.request_id,
                        status="completed",
                        result=resp_extract.model_dump(),
                        timing=resp_extract.timing,
                    )
                else:
                    raise ValueError(f"Unsupported endpoint: {dispatch.endpoint}")

            except Exception as exc:
                logger.error(f"Error executing job '{dispatch.job_id}': {exc}")
                job_result = WorkerJobResult(
                    job_id=dispatch.job_id,
                    request_id=dispatch.request_id,
                    status="failed",
                    error=str(exc),
                )

            # Send result back over WebSocket
            try:
                await ws.send(job_result.model_dump_json())
            except Exception as exc:
                logger.error(f"Failed to return result for '{dispatch.job_id}': {exc}")
