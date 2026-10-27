"""Job Dispatcher and Queue Manager for Gateway Control Plane with Caching and Scheduling."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from packages.caching.cache import InferenceCache
from packages.contracts.protocol import GatewayJobCancel, GatewayJobDispatch, WorkerJobResult
from packages.telemetry.logging import get_logger
from .registry import ConnectedWorker, WorkerRegistry
from .scheduler import SchedulingStrategy, WorkerScheduler

logger = get_logger("gateway_dispatcher")


class NoWorkerAvailableError(Exception):
    """Raised when no remote worker is registered or capable of processing the request."""


class QueueFullError(Exception):
    """Raised when the gateway pending job queue exceeds capacity."""


class JobTimeoutError(Exception):
    """Raised when a worker fails to return a result within the allotted time."""


class RemoteWorkerExecutionError(Exception):
    """Raised when the remote worker returns a failed status with an error message."""


class JobDispatcher:
    """Manages job dispatching, queueing, caching, and result resolution over WebSockets."""

    def __init__(
        self,
        registry: WorkerRegistry,
        max_queue_size: int = 100,
        default_timeout_sec: float = 60.0,
        cache: InferenceCache | None = None,
        scheduler: WorkerScheduler | None = None,
    ) -> None:
        self.registry = registry
        self.max_queue_size = max_queue_size
        self.default_timeout_sec = default_timeout_sec
        self.cache = cache
        self.scheduler = scheduler or WorkerScheduler(SchedulingStrategy.LEAST_LOADED)

        # job_id -> asyncio.Future[WorkerJobResult]
        self._pending_jobs: dict[str, asyncio.Future[WorkerJobResult]] = {}
        # job_id -> session_id
        self._job_worker_map: dict[str, str] = {}
        # Concurrency / Queue lock
        self._lock = asyncio.Lock()

    @property
    def pending_count(self) -> int:
        return len(self._pending_jobs)

    def _extract_cache_key(self, endpoint: str, payload: dict[str, Any], model_id: str | None) -> str | None:
        """Construct cache key if payload contains identifiable image and prompt."""
        if not self.cache:
            return None

        prompt = ""
        img_data = ""
        model = model_id or payload.get("model") or "unknown"

        if endpoint == "/v1/chat/completions":
            messages = payload.get("messages") or []
            texts = []
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                texts.append(part.get("text", ""))
                            elif part.get("type") == "image_url":
                                img_info = part.get("image_url", {})
                                if isinstance(img_info, dict):
                                    img_data = img_info.get("url", "")
            prompt = " ".join(texts)

        elif endpoint == "/v1/extract":
            prompt = payload.get("prompt") or payload.get("schema_name") or "extract"
            img_data = payload.get("image_url") or ""

        if not img_data and not prompt:
            return None

        return InferenceCache.generate_key(
            endpoint=endpoint,
            model=model,
            prompt=prompt,
            image_bytes_or_url=img_data,
            extra_params={"temp": payload.get("temperature", 0.0), "max_tokens": payload.get("max_tokens")},
        )

    async def dispatch_job(
        self,
        endpoint: str,
        payload: dict[str, Any],
        request_id: str,
        model_id: str | None = None,
        timeout_sec: float | None = None,
        wait_for_worker_sec: float = 2.0,
    ) -> dict[str, Any]:
        """Dispatch a job to an available remote worker or return cached result."""
        # 1. Check Inference Cache
        cache_key = self._extract_cache_key(endpoint, payload, model_id)
        if cache_key and self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache HIT for request '{request_id}' (key: {cache_key[:12]}...)")
                cached_result["cached"] = True
                return cached_result

        timeout = timeout_sec or self.default_timeout_sec

        async with self._lock:
            if len(self._pending_jobs) >= self.max_queue_size:
                logger.warning(f"Queue full ({len(self._pending_jobs)}/{self.max_queue_size}), rejecting {request_id}")
                raise QueueFullError("Gateway queue is full. Please try again later.")

        # 2. Acquire Worker using Scheduler policy
        worker = await self._acquire_worker(model_id=model_id, timeout_sec=wait_for_worker_sec)
        if not worker:
            logger.error(f"No worker available for model '{model_id}' (request: {request_id})")
            raise NoWorkerAvailableError(f"No available remote worker serving model '{model_id or 'any'}'")

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[WorkerJobResult] = loop.create_future()

        self._pending_jobs[job_id] = future
        self._job_worker_map[job_id] = worker.session_id

        # Mark worker busy
        worker.status = "busy"
        worker.active_job_id = job_id

        dispatch_msg = GatewayJobDispatch(
            job_id=job_id,
            request_id=request_id,
            endpoint=endpoint,
            payload=payload,
            timeout_sec=timeout,
        )

        logger.info(
            f"Dispatching job '{job_id}' to worker '{worker.worker_id}' (session: {worker.session_id}, endpoint: {endpoint})"
        )

        try:
            # Send dispatch message over WebSocket
            await worker.websocket.send_text(dispatch_msg.model_dump_json())
            logger.info(f"Successfully sent dispatch to worker '{worker.worker_id}'")

            # Wait for result with timeout
            job_result = await asyncio.wait_for(future, timeout=timeout)

            if job_result.status == "failed":
                raise RemoteWorkerExecutionError(job_result.error or "Remote worker failed without details")

            res_data = job_result.result or {}

            # Populate cache if successful
            if cache_key and self.cache:
                self.cache.set(cache_key, res_data)

            return res_data

        except asyncio.TimeoutError:
            logger.error(f"Job '{job_id}' timed out after {timeout}s on worker '{worker.worker_id}'")
            try:
                cancel_msg = GatewayJobCancel(job_id=job_id, reason="Gateway timeout exceeded")
                await worker.websocket.send_text(cancel_msg.model_dump_json())
            except Exception:
                pass
            raise JobTimeoutError(f"Worker did not respond within {timeout} seconds")

        finally:
            self._cleanup_job(job_id, worker)

    async def _acquire_worker(
        self,
        model_id: str | None = None,
        timeout_sec: float = 2.0,
    ) -> ConnectedWorker | None:
        """Attempt to get an eligible worker according to scheduling strategy."""
        start_time = time.time()
        while True:
            self.registry.prune_stale_workers()
            active_workers = list(self.registry.sessions.values())

            if self.scheduler:
                worker = self.scheduler.select_worker(active_workers, model_id=model_id)
            else:
                worker = self.registry.get_available_worker(model_id=model_id)

            if worker:
                return worker

            # If there are no workers at all registered, fail fast
            if self.registry.active_worker_count == 0:
                return None

            # If workers are registered but busy, wait up to timeout_sec
            if time.time() - start_time >= timeout_sec:
                return None

            await asyncio.sleep(0.05)

    def resolve_job(self, result: WorkerJobResult) -> bool:
        """Resolve a pending job future when a WorkerJobResult arrives."""
        future = self._pending_jobs.get(result.job_id)
        if not future or future.done():
            logger.warning(f"Received result for unknown or already completed job: {result.job_id}")
            return False

        future.set_result(result)
        logger.info(f"Job '{result.job_id}' resolved successfully with status: {result.status}")
        return True

    def handle_worker_disconnect(self, session_id: str) -> None:
        """Fail all pending jobs assigned to a disconnected worker."""
        orphaned_jobs = [
            jid for jid, sid in self._job_worker_map.items() if sid == session_id
        ]
        for jid in orphaned_jobs:
            future = self._pending_jobs.get(jid)
            if future and not future.done():
                logger.warning(f"Aborting job '{jid}' due to worker disconnect (session: {session_id})")
                future.set_exception(
                    NoWorkerAvailableError("Remote worker disconnected during job execution")
                )
            self._job_worker_map.pop(jid, None)
            self._pending_jobs.pop(jid, None)

    def _cleanup_job(self, job_id: str, worker: ConnectedWorker) -> None:
        """Clean up pending state and release worker back to idle."""
        self._pending_jobs.pop(job_id, None)
        self._job_worker_map.pop(job_id, None)

        if worker.session_id in self.registry.sessions:
            worker.status = "idle"
            worker.active_job_id = None
