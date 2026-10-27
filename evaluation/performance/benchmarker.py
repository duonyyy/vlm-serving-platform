"""Performance benchmarking: Latency percentiles, throughput, error rate, and VRAM."""

from __future__ import annotations

import statistics
import time
from typing import Any
from pydantic import BaseModel, Field

from packages.telemetry.timer import get_gpu_memory_stats


class PerformanceProfile(BaseModel):
    """Aggregated serving performance statistics."""

    attempted_requests: int
    completed_requests: int
    failed_requests: int
    error_rate_pct: float
    total_elapsed_seconds: float
    request_throughput_rps: float
    output_token_throughput_tps: float
    latency_e2e_ms: dict[str, float] = Field(description="mean, min, max, p50, p90, p95, p99")
    latency_inference_ms: dict[str, float]
    latency_preprocessing_ms: dict[str, float]
    peak_vram_allocated_bytes: int | None = None
    peak_vram_reserved_bytes: int | None = None


def calculate_percentiles(values: list[float]) -> dict[str, float]:
    """Compute summary statistics and quantiles."""
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

    s_vals = sorted(values)
    n = len(s_vals)

    def get_percentile(p: float) -> float:
        k = (n - 1) * p
        f = int(k)
        c = f + 1
        if c < n:
            return s_vals[f] + (k - f) * (s_vals[c] - s_vals[f])
        return s_vals[f]

    return {
        "mean": round(statistics.mean(s_vals), 2),
        "min": round(min(s_vals), 2),
        "max": round(max(s_vals), 2),
        "p50": round(get_percentile(0.50), 2),
        "p90": round(get_percentile(0.90), 2),
        "p95": round(get_percentile(0.95), 2),
        "p99": round(get_percentile(0.99), 2),
    }


def aggregate_performance_results(
    records: list[dict[str, Any]],
    total_elapsed_sec: float,
) -> PerformanceProfile:
    """Aggregate individual per-sample execution records into a formal PerformanceProfile."""
    attempted = len(records)
    completed = sum(1 for r in records if r.get("status") == "completed")
    failed = attempted - completed

    error_rate = round((failed / attempted) * 100.0, 2) if attempted > 0 else 0.0
    req_throughput = round(completed / total_elapsed_sec, 2) if total_elapsed_sec > 0 else 0.0

    total_out_tokens = sum(r.get("output_tokens", 0) for r in records if r.get("status") == "completed")
    token_throughput = round(total_out_tokens / total_elapsed_sec, 2) if total_elapsed_sec > 0 else 0.0

    e2e_latencies = [float(r["total_ms"]) for r in records if "total_ms" in r and r.get("status") == "completed"]
    infer_latencies = [float(r["inference_ms"]) for r in records if "inference_ms" in r and r.get("status") == "completed"]
    prep_latencies = [float(r["preprocessing_ms"]) for r in records if "preprocessing_ms" in r and r.get("status") == "completed"]

    gpu_stats = get_gpu_memory_stats()
    peak_alloc = gpu_stats.get("peak_allocated_bytes") if gpu_stats else None
    peak_res = gpu_stats.get("peak_reserved_bytes") if gpu_stats else None

    return PerformanceProfile(
        attempted_requests=attempted,
        completed_requests=completed,
        failed_requests=failed,
        error_rate_pct=error_rate,
        total_elapsed_seconds=round(total_elapsed_sec, 3),
        request_throughput_rps=req_throughput,
        output_token_throughput_tps=token_throughput,
        latency_e2e_ms=calculate_percentiles(e2e_latencies),
        latency_inference_ms=calculate_percentiles(infer_latencies),
        latency_preprocessing_ms=calculate_percentiles(prep_latencies),
        peak_vram_allocated_bytes=peak_alloc,
        peak_vram_reserved_bytes=peak_res,
    )
