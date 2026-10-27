"""Timing and hardware resource measurement utilities."""

from __future__ import annotations

import time
from typing import Any


class Timer:
    """Context manager for accurate wall-clock latency measurement in milliseconds."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed_ms = round((time.perf_counter() - self.start_time) * 1000, 3)


def get_gpu_memory_stats(device_index: int = 0) -> dict[str, int] | None:
    """Collect PyTorch GPU memory usage if CUDA is available, else return None."""
    try:
        import torch

        if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
            return None
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
        }
    except (ImportError, Exception):
        return None
