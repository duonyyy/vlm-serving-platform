"""CLI entry point to run the VLM Worker serving server."""

from __future__ import annotations

import argparse
import sys
import uvicorn

from packages.model_adapters.mock_adapter import MockAdapter
from packages.model_adapters.qwen_vl import QwenVLAdapter
from .app import create_app
from .manager import ModelManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM Serving Platform Worker API")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--adapter", type=str, choices=["mock", "qwen"], default="mock", help="Adapter engine to load")
    parser.add_argument("--model-id", type=str, default=None, help="Model ID override")
    parser.add_argument("--device", type=str, default=None, help="Device to use, e.g. cuda:0 or cpu")
    parser.add_argument("--revision", type=str, default=None, help="40-character commit revision SHA")
    args = parser.parse_args()

    manager = ModelManager()

    if args.adapter == "mock":
        model_id = args.model_id or "mock-vlm-2b"
        adapter = MockAdapter(model_id=model_id)
        manager.set_adapter(adapter)
    elif args.adapter == "qwen":
        model_id = args.model_id or "Qwen/Qwen3-VL-2B-Instruct"
        adapter = QwenVLAdapter(model_id=model_id)
        cfg: dict[str, str] = {}
        if args.device:
            cfg["device"] = args.device
        if args.revision:
            cfg["revision"] = args.revision
        manager.set_adapter(adapter, config=cfg)
    else:
        print(f"Unknown adapter: {args.adapter}", file=sys.stderr)
        sys.exit(1)

    app = create_app(manager)
    print(f"Starting VLM Worker on http://{args.host}:{args.port} with adapter [{args.adapter}]...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
