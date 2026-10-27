"""CLI Entry point for launching the Gateway Server."""

from __future__ import annotations

import argparse
import sys
import uvicorn

from apps.gateway.config import GatewayConfig
from apps.gateway.app import create_gateway_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM Serving Gateway Control Plane")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--token", type=str, default=None, help="Worker authentication token")
    parser.add_argument("--client-key", type=str, default=None, help="Optional client API key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GatewayConfig(
        host=args.host,
        port=args.port,
        worker_auth_token=args.token or "vlm_worker_secret_2026",
        client_api_key=args.client_key,
    )
    gateway_app = create_gateway_app(config)
    print(f"[*] Starting VLM Gateway on http://{config.host}:{config.port}")
    print(f"[*] Worker registration WebSocket endpoint: ws://{config.host}:{config.port}/workers/connect")
    uvicorn.run(gateway_app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
