"""Prometheus Metrics Exporter for VLM Gateway and Worker Runtime."""

from __future__ import annotations

import threading
import time
from typing import Dict, List


class PrometheusMetrics:
    """Thread-safe Prometheus metrics collector and text exposition formatter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Counters: (endpoint, model, status_code) -> count
        self._requests_total: Dict[tuple[str, str, int], int] = {}

        # Latency buckets (seconds): [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, +Inf]
        self._buckets = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        # (endpoint, model) -> bucket_counts
        self._duration_buckets: Dict[tuple[str, str], List[int]] = {}
        self._duration_sum: Dict[tuple[str, str], float] = {}
        self._duration_count: Dict[tuple[str, str], int] = {}

        # Gauges
        self._active_workers: int = 0
        self._pending_jobs: int = 0
        self._gpu_vram_used_bytes: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def record_request(self, endpoint: str, model: str, status_code: int, duration_sec: float) -> None:
        """Record an incoming request completion and its latency."""
        with self._lock:
            # 1. Total count
            key = (endpoint, model, status_code)
            self._requests_total[key] = self._requests_total.get(key, 0) + 1

            # 2. Histogram
            hist_key = (endpoint, model)
            if hist_key not in self._duration_buckets:
                self._duration_buckets[hist_key] = [0] * (len(self._buckets) + 1)
                self._duration_sum[hist_key] = 0.0
                self._duration_count[hist_key] = 0

            self._duration_sum[hist_key] += duration_sec
            self._duration_count[hist_key] += 1

            for idx, bound in enumerate(self._buckets):
                if duration_sec <= bound:
                    self._duration_buckets[hist_key][idx] += 1
            # +Inf bucket
            self._duration_buckets[hist_key][-1] += 1

    def set_active_workers(self, count: int) -> None:
        with self._lock:
            self._active_workers = max(0, count)

    def set_pending_jobs(self, count: int) -> None:
        with self._lock:
            self._pending_jobs = max(0, count)

    def set_gpu_vram(self, bytes_used: int) -> None:
        with self._lock:
            self._gpu_vram_used_bytes = max(0, bytes_used)

    def record_cache(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def render_prometheus_text(self) -> str:
        """Render standard Prometheus exposition format."""
        lines = []

        with self._lock:
            # Gauges
            lines.append("# HELP vlm_active_workers Current number of registered active remote workers.")
            lines.append("# TYPE vlm_active_workers gauge")
            lines.append(f"vlm_active_workers {self._active_workers}")

            lines.append("# HELP vlm_pending_jobs Current number of pending jobs in queue.")
            lines.append("# TYPE vlm_pending_jobs gauge")
            lines.append(f"vlm_pending_jobs {self._pending_jobs}")

            lines.append("# HELP vlm_gpu_vram_used_bytes Reported GPU VRAM allocation in bytes.")
            lines.append("# TYPE vlm_gpu_vram_used_bytes gauge")
            lines.append(f"vlm_gpu_vram_used_bytes {self._gpu_vram_used_bytes}")

            lines.append("# HELP vlm_cache_hits_total Total count of cache hits.")
            lines.append("# TYPE vlm_cache_hits_total counter")
            lines.append(f"vlm_cache_hits_total {self._cache_hits}")

            lines.append("# HELP vlm_cache_misses_total Total count of cache misses.")
            lines.append("# TYPE vlm_cache_misses_total counter")
            lines.append(f"vlm_cache_misses_total {self._cache_misses}")

            # Requests counter
            lines.append("# HELP vlm_requests_total Total number of completed requests.")
            lines.append("# TYPE vlm_requests_total counter")
            for (endpoint, model, status_code), count in self._requests_total.items():
                lines.append(
                    f'vlm_requests_total{{endpoint="{endpoint}",model="{model}",status_code="{status_code}"}} {count}'
                )

            # Latency histogram
            lines.append("# HELP vlm_request_duration_seconds Request duration in seconds.")
            lines.append("# TYPE vlm_request_duration_seconds histogram")
            for (endpoint, model), bucket_counts in self._duration_buckets.items():
                running_total = 0
                for idx, bound in enumerate(self._buckets):
                    running_total = bucket_counts[idx]
                    lines.append(
                        f'vlm_request_duration_seconds_bucket{{endpoint="{endpoint}",model="{model}",le="{bound}"}} {running_total}'
                    )
                lines.append(
                    f'vlm_request_duration_seconds_bucket{{endpoint="{endpoint}",model="{model}",le="+Inf"}} {bucket_counts[-1]}'
                )
                lines.append(
                    f'vlm_request_duration_seconds_sum{{endpoint="{endpoint}",model="{model}"}} {self._duration_sum[(endpoint, model)]:.4f}'
                )
                lines.append(
                    f'vlm_request_duration_seconds_count{{endpoint="{endpoint}",model="{model}"}} {self._duration_count[(endpoint, model)]}'
                )

        return "\n".join(lines) + "\n"


# Global singleton instance for platform-wide metrics
global_metrics = PrometheusMetrics()
