"""Integration tests for Load Testing and SLO verification."""

import asyncio
import json
import socket
import tempfile
import threading
import time
import unittest
import httpx
import uvicorn
import websockets

from apps.gateway.app import create_gateway_app
from apps.gateway.config import GatewayConfig
from evaluation.load_test_runner import LoadTestRunner
from packages.contracts.chat import ChatChoice, ChatCompletionResponse, ChatResponseMessage, TimingBreakdown, UsageInfo
from packages.contracts.protocol import WorkerJobResult, WorkerRegister


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestLoadAndSLO(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.port = get_free_port()
        cls.token = "load_secret_token"
        cls.config = GatewayConfig(worker_auth_token=cls.token, job_timeout_sec=5.0)
        cls.app = create_gateway_app(cls.config)
        server_config = uvicorn.Config(cls.app, host="127.0.0.1", port=cls.port, log_level="error")
        cls.server = uvicorn.Server(server_config)
        cls.server_thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.server_thread.start()
        time.sleep(0.4)

        # Start mock worker responding to jobs
        cls.stop_worker = threading.Event()
        cls.worker_ready = threading.Event()

        def worker_loop():
            async def run():
                async with websockets.connect(f"ws://127.0.0.1:{cls.port}/workers/connect") as ws:
                    reg = WorkerRegister(worker_id="load_w", model_id="Qwen/Qwen2-VL-2B-Instruct", token=cls.token)
                    await ws.send(reg.model_dump_json())
                    await ws.recv()
                    cls.worker_ready.set()

                    while not cls.stop_worker.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        except asyncio.TimeoutError:
                            continue
                        except Exception:
                            break
                        msg = json.loads(raw)
                        if msg.get("type") == "job_dispatch":
                            resp = ChatCompletionResponse(
                                id=msg["request_id"],
                                model="Qwen/Qwen2-VL-2B-Instruct",
                                choices=[ChatChoice(message=ChatResponseMessage(content="SLO check OK"))],
                                usage=UsageInfo(total_tokens=10),
                                timing=TimingBreakdown(total_ms=5.0),
                            )
                            res = WorkerJobResult(
                                job_id=msg["job_id"],
                                request_id=msg["request_id"],
                                status="completed",
                                result=resp.model_dump(),
                            )
                            await ws.send(res.model_dump_json())

            asyncio.run(run())

        cls.wt = threading.Thread(target=worker_loop, daemon=True)
        cls.wt.start()
        cls.worker_ready.wait(timeout=3.0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.stop_worker.set()
        cls.server.should_exit = True
        cls.wt.join(timeout=1.0)
        cls.server_thread.join(timeout=1.0)

    def test_load_test_and_slo_compliance(self) -> None:
        """Run concurrent load test and verify metrics export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = LoadTestRunner(
                base_url=f"http://127.0.0.1:{self.port}",
                total_requests=10,
                concurrency=3,
                reports_dir=tmpdir,
            )
            summary = asyncio.run(runner.run_load_test())

            self.assertEqual(summary["total_requests"], 10)
            self.assertEqual(summary["successful_requests"], 10)
            self.assertEqual(summary["error_rate_pct"], 0.0)
            self.assertGreater(summary["throughput_rps"], 0)
            self.assertTrue(summary["slo_eval"]["overall_pass"])

            # Verify Prometheus /metrics endpoint exposes telemetry
            client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}")
            metrics_resp = client.get("/metrics")
            self.assertEqual(metrics_resp.status_code, 200)
            self.assertIn("vlm_requests_total", metrics_resp.text)
            self.assertIn("vlm_active_workers", metrics_resp.text)


if __name__ == "__main__":
    unittest.main()
