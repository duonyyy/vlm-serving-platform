"""Unit tests for model adapter lifecycles and behavior."""

import unittest
from PIL import Image
from packages.model_adapters.mock_adapter import MockAdapter
from packages.model_adapters.qwen_vl import QwenVLAdapter


class ModelAdapterTests(unittest.TestCase):
    def test_mock_adapter_lifecycle(self) -> None:
        adapter = MockAdapter("test-vlm")
        self.assertEqual(adapter.status, "NOT_LOADED")

        # Inference before load must fail
        with self.assertRaises(RuntimeError) as ctx:
            adapter.infer("test prompt")
        self.assertIn("not in READY state", str(ctx.exception))

        # Load
        adapter.load()
        self.assertEqual(adapter.status, "READY")
        self.assertIsNone(adapter.health()["error"])

        # Infer with prompt and image
        test_img = Image.new("RGB", (300, 200), color="white")
        result = adapter.infer("Mô tả ảnh này", image=test_img)
        self.assertTrue(len(result.prediction_text) > 0)
        self.assertIn("300x200", result.prediction_text)
        self.assertGreater(result.input_tokens, 0)
        self.assertGreater(result.output_tokens, 0)
        self.assertEqual(result.finish_reason, "stop")

        # Unload
        adapter.unload()
        self.assertEqual(adapter.status, "NOT_LOADED")

    def test_mock_adapter_error_simulation(self) -> None:
        # Load failure simulation
        adapter = MockAdapter()
        with self.assertRaises(RuntimeError) as ctx:
            adapter.load({"fail_on_load": True})
        self.assertEqual(adapter.status, "ERROR")

        # OOM simulation
        adapter2 = MockAdapter()
        adapter2.load({"simulate_oom": True})
        with self.assertRaises(RuntimeError) as ctx:
            adapter2.infer("prompt")
        self.assertIn("CUDA out of memory", str(ctx.exception))

        # Prompt-triggered failure
        adapter3 = MockAdapter()
        adapter3.load({"fail_on_prompt": "CRASH"})
        with self.assertRaises(ValueError):
            adapter3.infer("Please CRASH now")

    def test_qwen_adapter_initialization(self) -> None:
        adapter = QwenVLAdapter(model_id="Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(adapter.status, "NOT_LOADED")
        self.assertEqual(adapter.model_id, "Qwen/Qwen3-VL-2B-Instruct")
        self.assertTrue(adapter.capabilities.image_chat)

        # Invalid revision format check
        with self.assertRaises(RuntimeError) as ctx:
            adapter.load({"revision": "invalid_non_hex_sha"})
        self.assertEqual(adapter.status, "ERROR")


if __name__ == "__main__":
    unittest.main()
