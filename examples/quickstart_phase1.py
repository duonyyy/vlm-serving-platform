"""Quickstart example demonstrating Phase 1 VLM Serving & Client SDK."""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from PIL import Image

from client import VLMClient
from packages.model_adapters import MockAdapter
from packages.worker_runtime import ModelManager, create_app


def main() -> None:
    print("=" * 60)
    print("VLM Serving Platform — Phase 1 Quickstart Demonstration")
    print("=" * 60)

    # 1. Initialize Worker Manager with MockAdapter
    print("\n[1] Initializing Worker Runtime...")
    manager = ModelManager(worker_id="worker_quickstart_01")
    adapter = MockAdapter(model_id="qwen3-vl-2b")
    manager.set_adapter(adapter)
    app = create_app(manager)

    # 2. Initialize Client SDK (in-process for demo)
    test_client = TestClient(app)
    client = VLMClient(base_url="http://localhost:8000", http_client=test_client)

    # 3. Check Worker Health
    print("\n[2] Checking Worker Health...")
    health = client.get_health()
    print(f"  Status: {health.status}")
    print(f"  Worker ID: {health.worker_id}")
    print(f"  Active Model: {health.active_model} ({health.active_model_status})")
    print(f"  CUDA Available: {health.cuda_available}")

    # 4. List Models
    print("\n[3] Listing Models...")
    models = client.list_models()
    for m in models:
        print(f"  Model ID: {m.id} | Capabilities: {m.capabilities.model_dump()}")

    # 5. Create a synthetic test image
    print("\n[4] Generating sample test image in memory...")
    sample_img = Image.new("RGB", (640, 480), color="#1A365D")

    # 6. Multimodal Chat Completion
    print("\n[5] Sending Multimodal Chat Completion Request...")
    chat_resp = client.chat_image(
        image=sample_img,
        prompt="Mô tả các chi tiết nổi bật trong bức ảnh này.",
        model="qwen3-vl-2b",
        request_id="req_quickstart_chat",
    )
    print(f"  Response ID: {chat_resp.id}")
    print(f"  Model: {chat_resp.model}")
    print(f"  Prediction: {chat_resp.choices[0].message.content}")
    print(f"  Usage: {chat_resp.usage.total_tokens} tokens (in: {chat_resp.usage.prompt_tokens}, out: {chat_resp.usage.completion_tokens})")
    if chat_resp.timing:
        print(f"  Latency: Preprocessing={chat_resp.timing.preprocessing_ms}ms, Inference={chat_resp.timing.inference_ms}ms, Total={chat_resp.timing.total_ms}ms")

    # 7. Structured Invoice Extraction
    print("\n[6] Sending Structured Invoice Extraction Request...")
    extract_resp = client.extract(
        image=sample_img,
        schema_name="invoice",
        model="qwen3-vl-2b",
        request_id="req_quickstart_extract",
    )
    print(f"  Status: {extract_resp.status}")
    print(f"  Extracted Data: {extract_resp.data}")
    if extract_resp.timing:
        print(f"  Total Duration: {extract_resp.timing.total_ms}ms")

    print("\n" + "=" * 60)
    print("Phase 1 Quickstart completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
