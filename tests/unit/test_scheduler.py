"""Unit tests for WorkerScheduler and load balancing strategies."""

import time
import unittest

from apps.gateway.registry import ConnectedWorker
from apps.gateway.scheduler import SchedulingStrategy, WorkerScheduler
from packages.contracts.models import ModelCapabilities


class TestWorkerScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.workers = [
            ConnectedWorker(
                session_id="s1",
                worker_id="w1",
                model_id="Qwen/Qwen2-VL-2B-Instruct",
                capabilities=ModelCapabilities(),
                last_heartbeat=time.time() - 10.0,
            ),
            ConnectedWorker(
                session_id="s2",
                worker_id="w2",
                model_id="Qwen/Qwen2-VL-2B-Instruct",
                capabilities=ModelCapabilities(),
                last_heartbeat=time.time() - 2.0,
            ),
            ConnectedWorker(
                session_id="s3",
                worker_id="w3",
                model_id="PaddleOCR-VL-1.6",
                capabilities=ModelCapabilities(),
                status="busy",  # not eligible
            ),
        ]

    def test_round_robin_selection(self) -> None:
        scheduler = WorkerScheduler(SchedulingStrategy.ROUND_ROBIN)
        chosen1 = scheduler.select_worker(self.workers)
        chosen2 = scheduler.select_worker(self.workers)
        self.assertIsNotNone(chosen1)
        self.assertIsNotNone(chosen2)
        # Should cycle between eligible idle workers (w1 and w2)
        self.assertNotEqual(chosen1.worker_id, chosen2.worker_id)

    def test_least_loaded_selection(self) -> None:
        scheduler = WorkerScheduler(SchedulingStrategy.LEAST_LOADED)
        chosen = scheduler.select_worker(self.workers)
        # w1 has older last_heartbeat (-10s) compared to w2 (-2s)
        self.assertEqual(chosen.worker_id, "w1")

    def test_model_affinity_filtering(self) -> None:
        scheduler = WorkerScheduler(SchedulingStrategy.MODEL_AFFINITY)
        chosen = scheduler.select_worker(self.workers, model_id="Qwen/Qwen2-VL-2B-Instruct")
        self.assertIn(chosen.worker_id, ["w1", "w2"])

        # No idle worker serves PaddleOCR
        no_worker = scheduler.select_worker(self.workers, model_id="PaddleOCR-VL-1.6")
        self.assertIsNone(no_worker)


if __name__ == "__main__":
    unittest.main()
