"""Fault Injection and Resilience Test Suite for Phase 2 Gateway."""

import asyncio
import json
import socket
import threading
import time
import unittest
import httpx
import uvicorn
import websockets

from apps.gateway.app import create_gateway_app
from apps.gateway.config import GatewayConfig
from packages.contracts.chat import ChatChoice, ChatCompletionResponse, ChatResponseMessage, TimingBreakdown, UsageInfo
from packages.contracts.protocol import WorkerJobResult, WorkerRegister


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestFaultInjectionResilience(unittest.TestCase):
    def test_no_worker_available_returns_503(self) -> None:
        """When no remote worker is registered, Gateway must return 503."""
        port = get_free_port()
        config = GatewayConfig(worker_auth_token="token_test")
        app = create_gateway_app(config)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        time.sleep(0.3)

        try:
            client = httpx.Client(base_url=f"http://127.0.0.1:{port}")
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "Qwen/Qwen2-VL-2B-Instruct", "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(resp.status_code, 503)
            data = resp.json()
            self.assertEqual(data["error"]["type"], "ServiceUnavailableError")
            self.assertEqual(data["error"]["code"], "no_workers_available")
            self.assertIn("request_id", data)
        finally:
            server.should_exit = True
            t.join(timeout=1.0)

    def test_queue_full_returns_429(self) -> None:
        """When queue capacity is exceeded, Gateway must return 429."""
        port = get_free_port()
        config = GatewayConfig(worker_auth_token="token_test", queue_max_size=1, job_timeout_sec=3.0)
        app = create_gateway_app(config)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        time.sleep(0.3)

        worker_ready = threading.Event()
        release_worker = threading.Event()

        def sleepy_worker() -> None:
            async def run():
                async with websockets.connect(f"ws://127.0.0.1:{port}/workers/connect") as ws:
                    reg = WorkerRegister(worker_id="w1", model_id="Qwen/Qwen2-VL-2B-Instruct", token="token_test")
                    await ws.send(reg.model_dump_json())
                    await ws.recv()
                    worker_ready.set()

                    # Block on first job until released
                    await ws.recv()
                    while not release_worker.is_set():
                        await asyncio.sleep(0.05)

            asyncio.run(run())

        wt = threading.Thread(target=sleepy_worker, daemon=True)
        wt.start()
        self.assertTrue(worker_ready.wait(timeout=2.0))

        client = httpx.Client(base_url=f"http://127.0.0.1:{port}")

        try:
            # Job 1 starts in background
            def send_job_1():
                try:
                    client.post(
                        "/v1/chat/completions",
                        json={"model": "Qwen/Qwen2-VL-2B-Instruct", "messages": [{"role": "user", "content": "1"}]},
                        timeout=5.0,
                    )
                except Exception:
                    pass

            t_job1 = threading.Thread(target=send_job_1, daemon=True)
            t_job1.start()
            time.sleep(0.15)  # Ensure job 1 is dispatched to worker and occupies queue/worker

            # Job 2 should exceed queue capacity and immediately get 429
            resp2 = client.post(
                "/v1/chat/completions",
                json={"model": "Qwen/Qwen2-VL-2B-Instruct", "messages": [{"role": "user", "content": "2"}]},
            )
            self.assertEqual(resp2.status_code, 429)
            self.assertEqual(resp2.json()["error"]["type"], "RateLimitError")
        finally:
            release_worker.set()
            server.should_exit = True
            wt.join(timeout=1.0)
            t.join(timeout=1.0)

    def test_job_timeout_returns_504(self) -> None:
        """When worker exceeds job_timeout_sec, Gateway must return 504."""
        port = get_free_port()
        config = GatewayConfig(worker_auth_token="token_test", job_timeout_sec=0.25)
        app = create_gateway_app(config)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        time.sleep(0.3)

        worker_ready = threading.Event()

        def unresponsive_worker() -> None:
            async def run():
                async with websockets.connect(f"ws://127.0.0.1:{port}/workers/connect") as ws:
                    reg = WorkerRegister(worker_id="unresp_w", model_id="Qwen/Qwen2-VL-2B-Instruct", token="token_test")
                    await ws.send(reg.model_dump_json())
                    await ws.recv()
                    worker_ready.set()

                    # Receive job but sleep past timeout
                    await ws.recv()
                    await asyncio.sleep(0.6)

            asyncio.run(run())

        wt = threading.Thread(target=unresponsive_worker, daemon=True)
        wt.start()
        self.assertTrue(worker_ready.wait(timeout=2.0))

        try:
            client = httpx.Client(base_url=f"http://127.0.0.1:{port}")
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "Qwen/Qwen2-VL-2B-Instruct", "messages": [{"role": "user", "content": "timeout test"}]},
            )
            self.assertEqual(resp.status_code, 504)
            data = resp.json()
            self.assertEqual(data["error"]["type"], "TimeoutError")
            self.assertEqual(data["error"]["code"], "worker_timeout")
        finally:
            server.should_exit = True
            wt.join(timeout=1.0)
            t.join(timeout=1.0)

    def test_worker_disconnect_during_job(self) -> None:
        """When worker abruptly disconnects during job execution, request fails gracefully with 503."""
        port = get_free_port()
        config = GatewayConfig(worker_auth_token="token_test", job_timeout_sec=2.0)
        app = create_gateway_app(config)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        time.sleep(0.3)

        worker_ready = threading.Event()

        def disconnecting_worker() -> None:
            async def run():
                async with websockets.connect(f"ws://127.0.0.1:{port}/workers/connect") as ws:
                    reg = WorkerRegister(worker_id="disc_w", model_id="Qwen/Qwen2-VL-2B-Instruct", token="token_test")
                    await ws.send(reg.model_dump_json())
                    await ws.recv()
                    worker_ready.set()

                    # Wait for job dispatch then abruptly close connection
                    await ws.recv()
                    await ws.close()

            asyncio.run(run())

        wt = threading.Thread(target=disconnecting_worker, daemon=True)
        wt.start()
        self.assertTrue(worker_ready.wait(timeout=2.0))

        try:
            client = httpx.Client(base_url=f"http://127.0.0.1:{port}")
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "Qwen/Qwen2-VL-2B-Instruct", "messages": [{"role": "user", "content": "disc test"}]},
            )
            self.assertEqual(resp.status_code, 503)
            self.assertEqual(resp.json()["error"]["type"], "ServiceUnavailableError")
        finally:
            server.should_exit = True
            wt.join(timeout=1.0)
            t.join(timeout=1.0)

    def test_worker_reconnect_success(self) -> None:
        """When worker disconnects and reconnects, subsequent jobs succeed without issue."""
        port = get_free_port()
        config = GatewayConfig(worker_auth_token="token_test", job_timeout_sec=3.0)
        app = create_gateway_app(config)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        time.sleep(0.3)

        # 1. Connect and immediately disconnect
        async def quick_connect_disconnect():
            async with websockets.connect(f"ws://127.0.0.1:{port}/workers/connect") as ws:
                reg = WorkerRegister(worker_id="recon_w", model_id="Qwen/Qwen2-VL-2B-Instruct", token="token_test")
                await ws.send(reg.model_dump_json())
                await ws.recv()
                await ws.close()

        asyncio.run(quick_connect_disconnect())
        time.sleep(0.1)

        # 2. Reconnect with a new session and handle a real job
        worker_ready = threading.Event()

        def reconnected_worker():
            async def run():
                async with websockets.connect(f"ws://127.0.0.1:{port}/workers/connect") as ws:
                    reg = WorkerRegister(worker_id="recon_w", model_id="Qwen/Qwen2-VL-2B-Instruct", token="token_test")
                    await ws.send(reg.model_dump_json())
                    await ws.recv()
                    worker_ready.set()

                    raw = await ws.recv()
                    msg = json.loads(raw)
                    if msg.get("type") == "job_dispatch":
                        resp_data = ChatCompletionResponse(
                            id=msg["request_id"],
                            model="Qwen/Qwen2-VL-2B-Instruct",
                            choices=[
                                ChatChoice(
                                    index=0,
                                    message=ChatResponseMessage(role="assistant", content="Reconnected and served!"),
                                )
                            ],
                            usage=UsageInfo(total_tokens=10),
                            timing=TimingBreakdown(total_ms=15.0),
                        )
                        res = WorkerJobResult(
                            job_id=msg["job_id"],
                            request_id=msg["request_id"],
                            status="completed",
                            result=resp_data.model_dump(),
                        )
                        await ws.send(res.model_dump_json())

            asyncio.run(run())

        wt = threading.Thread(target=reconnected_worker, daemon=True)
        wt.start()
        self.assertTrue(worker_ready.wait(timeout=2.0))

        try:
            client = httpx.Client(base_url=f"http://127.0.0.1:{port}")
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "Qwen/Qwen2-VL-2B-Instruct", "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["choices"][0]["message"]["content"], "Reconnected and served!")
        finally:
            server.should_exit = True
            wt.join(timeout=1.0)
            t.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
