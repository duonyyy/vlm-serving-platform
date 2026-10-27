"""Integration tests for FastAPI Worker Runtime API."""

import base64
import io
import unittest
from fastapi.testclient import TestClient
from PIL import Image

from packages.model_adapters.mock_adapter import MockAdapter
from packages.worker_runtime.app import create_app
from packages.worker_runtime.manager import ModelManager


def make_b64_png(size: tuple[int, int] = (64, 64)) -> str:
    img = Image.new("RGB", size, color="teal")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class WorkerApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ModelManager(worker_id="test_worker_1")
        self.adapter = MockAdapter(model_id="test-mock-vlm")
        self.manager.set_adapter(self.adapter)
        self.app = create_app(self.manager)
        self.client = TestClient(self.app)

    def test_health_endpoint(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["active_model"], "test-mock-vlm")
        self.assertIn("X-Request-ID", resp.headers)
        self.assertIn("X-Process-Time-Ms", resp.headers)

    def test_list_and_get_models(self) -> None:
        resp = self.client.get("/v1/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["id"], "test-mock-vlm")

        # Specific model
        resp2 = self.client.get("/v1/models/test-mock-vlm")
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp2.json()["id"], "test-mock-vlm")

        # Non-existent model
        resp3 = self.client.get("/v1/models/non-existent")
        self.assertEqual(resp3.status_code, 404)

    def test_chat_completions_with_image(self) -> None:
        img_uri = make_b64_png((100, 100))
        payload = {
            "model": "test-mock-vlm",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Mô tả bức ảnh này"},
                        {"type": "image_url", "image_url": {"url": img_uri}},
                    ],
                }
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        }
        resp = self.client.post("/v1/chat/completions", json=payload, headers={"X-Request-ID": "req_custom_001"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("X-Request-ID"), "req_custom_001")
        data = resp.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["model"], "test-mock-vlm")
        self.assertTrue(len(data["choices"]) > 0)
        self.assertIn("100x100", data["choices"][0]["message"]["content"])
        self.assertGreater(data["usage"]["total_tokens"], 0)
        self.assertIn("timing", data)

    def test_chat_completions_text_only(self) -> None:
        payload = {
            "model": "test-mock-vlm",
            "messages": [{"role": "user", "content": "Xin chào VLM"}],
            "max_tokens": 64,
        }
        resp = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(len(data["choices"][0]["message"]["content"]) > 0)

    def test_chat_completions_corrupt_image_returns_400(self) -> None:
        payload = {
            "model": "test-mock-vlm",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Đọc ảnh"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,not_valid_image_bytes_here"}},
                    ],
                }
            ],
        }
        resp = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertEqual(data["error"]["type"], "invalid_request_error")
        self.assertIn("request_id", data)

    def test_chat_completions_when_model_unready_returns_503(self) -> None:
        # Create worker with unready model
        empty_mgr = ModelManager()
        empty_app = create_app(empty_mgr)
        client = TestClient(empty_app)

        payload = {
            "model": "test-mock-vlm",
            "messages": [{"role": "user", "content": "Test"}],
        }
        resp = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(resp.status_code, 503)
        data = resp.json()
        self.assertEqual(data["error"]["type"], "runtime_error")

    def test_extract_endpoint(self) -> None:
        img_uri = make_b64_png((80, 80))
        payload = {
            "model": "test-mock-vlm",
            "image_url": img_uri,
            "schema_name": "invoice",
        }
        resp = self.client.post("/v1/extract", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["data"].get("seller_name"), "Công ty TNHH Cà Phê Việt")


if __name__ == "__main__":
    unittest.main()
