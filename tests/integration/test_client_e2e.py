"""End-to-End integration tests for VLM Python Client SDK against the Worker API."""

import io
import unittest
import httpx
from PIL import Image

from client.vlm_client import VLMAPIError, VLMClient
from packages.model_adapters.mock_adapter import MockAdapter
from packages.worker_runtime.app import create_app
from packages.worker_runtime.manager import ModelManager


class ClientE2EIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ModelManager(worker_id="e2e_worker")
        self.adapter = MockAdapter(model_id="qwen3-vl-2b")
        self.manager.set_adapter(self.adapter)
        self.app = create_app(self.manager)

        from fastapi.testclient import TestClient
        self.test_http_client = TestClient(self.app)
        self.client = VLMClient(base_url="http://testserver", http_client=self.test_http_client)

    def test_client_health_and_models(self) -> None:
        health = self.client.get_health()
        self.assertEqual(health.status, "healthy")
        self.assertEqual(health.active_model, "qwen3-vl-2b")

        models = self.client.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "qwen3-vl-2b")

    def test_client_chat_image_pillow(self) -> None:
        img = Image.new("RGB", (250, 150), color="coral")

        # Chat with image directly passing PIL.Image object
        response = self.client.chat_image(
            image=img,
            prompt="Mô tả bức ảnh này",
            model="qwen3-vl-2b",
            request_id="req_e2e_img_test",
        )

        self.assertEqual(response.model, "qwen3-vl-2b")
        self.assertTrue(len(response.choices) > 0)
        self.assertIn("250x150", response.choices[0].message.content)
        self.assertGreater(response.usage.total_tokens, 0)
        self.assertIsNotNone(response.timing)
        self.assertGreater(response.timing.total_ms, 0)

    def test_client_extract_invoice(self) -> None:
        img = Image.new("RGB", (120, 120), color="white")
        res = self.client.extract(image=img, schema_name="invoice")
        self.assertEqual(res.status, "completed")
        self.assertEqual(res.data.get("seller_name"), "Công ty TNHH Cà Phê Việt")
        self.assertEqual(res.data.get("currency"), "VND")

    def test_client_handles_error_response(self) -> None:
        # Pass corrupt base64
        with self.assertRaises(VLMAPIError) as ctx:
            self.client.chat_image(
                image="data:image/jpeg;base64,invalid_corrupt_data",
                prompt="Test fail",
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.error_type, "invalid_request_error")


if __name__ == "__main__":
    unittest.main()
