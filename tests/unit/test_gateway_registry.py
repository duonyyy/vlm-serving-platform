"""Unit tests for Gateway WorkerRegistry."""

import time
import unittest
from unittest.mock import MagicMock

from apps.gateway.registry import WorkerRegistry
from packages.contracts.models import ModelCapabilities
from packages.contracts.protocol import WorkerHeartbeat, WorkerRegister


class TestGatewayRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.auth_token = "secret_token_123"
        self.registry = WorkerRegistry(auth_token=self.auth_token, heartbeat_ttl_sec=0.2)

    def test_register_success(self) -> None:
        mock_ws = MagicMock()
        reg = WorkerRegister(
            worker_id="worker_01",
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            token=self.auth_token,
            capabilities=ModelCapabilities(image_input=True),
        )
        success, sid, reason = self.registry.register(reg, mock_ws)
        self.assertTrue(success)
        self.assertTrue(sid.startswith("sess_"))
        self.assertIsNone(reason)
        self.assertEqual(self.registry.active_worker_count, 1)

    def test_register_invalid_token(self) -> None:
        mock_ws = MagicMock()
        reg = WorkerRegister(
            worker_id="worker_02",
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            token="wrong_token",
        )
        success, sid, reason = self.registry.register(reg, mock_ws)
        self.assertFalse(success)
        self.assertEqual(sid, "")
        self.assertIn("Authentication failed", reason or "")
        self.assertEqual(self.registry.active_worker_count, 0)

    def test_record_heartbeat(self) -> None:
        mock_ws = MagicMock()
        reg = WorkerRegister(
            worker_id="worker_01",
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            token=self.auth_token,
        )
        _, sid, _ = self.registry.register(reg, mock_ws)

        hb = WorkerHeartbeat(worker_id="worker_01", session_id=sid, status="busy")
        updated = self.registry.record_heartbeat(hb)
        self.assertTrue(updated)
        worker = self.registry.sessions[sid]
        self.assertEqual(worker.status, "busy")

    def test_prune_stale_workers(self) -> None:
        mock_ws = MagicMock()
        reg = WorkerRegister(
            worker_id="worker_stale",
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            token=self.auth_token,
        )
        _, sid, _ = self.registry.register(reg, mock_ws)
        self.assertEqual(self.registry.active_worker_count, 1)

        # Wait for TTL expiry
        time.sleep(0.25)
        self.assertEqual(self.registry.active_worker_count, 0)
        self.assertNotIn(sid, self.registry.sessions)

    def test_list_models(self) -> None:
        mock_ws = MagicMock()
        reg = WorkerRegister(
            worker_id="worker_01",
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            token=self.auth_token,
            device_info={"name": "Tesla T4"},
        )
        self.registry.register(reg, mock_ws)

        models = self.registry.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(models[0].device, "Tesla T4")


if __name__ == "__main__":
    unittest.main()
