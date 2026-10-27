"""Asynchronous Load Testing and SLO Verification Suite for Production VLM Serving."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any
import httpx

from packages.telemetry.logging import get_logger

logger = get_logger("load_tester")


class SLOCriteria:
    """Service Level Objectives thresholds for production deployment."""

    MAX_P95_LATENCY_MS: float = 1500.0
    MAX_ERROR_RATE_PCT: float = 1.0
    MIN_AVAILABILITY_PCT: float = 99.0
    MIN_THROUGHPUT_RPS: float = 5.0


class LoadTestRunner:
    """Generates concurrent synthetic traffic against Gateway and evaluates SLO compliance."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        total_requests: int = 50,
        concurrency: int = 5,
        reports_dir: str = "reports",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.total_requests = total_requests
        self.concurrency = concurrency
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def _send_single_request(
        self, client: httpx.AsyncClient, req_id: int
    ) -> dict[str, Any]:
        """Send one chat completion request and measure turnaround time."""
        payload = {
            "model": "Qwen/Qwen2-VL-2B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": f"Kiểm tra tính hợp lệ của hóa đơn số {req_id}",
                }
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=30.0,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            is_success = resp.status_code == 200
            return {
                "req_id": req_id,
                "status_code": resp.status_code,
                "success": is_success,
                "latency_ms": elapsed_ms,
                "error": None if is_success else resp.text,
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "req_id": req_id,
                "status_code": 0,
                "success": False,
                "latency_ms": elapsed_ms,
                "error": str(exc),
            }

    async def run_load_test(self) -> dict[str, Any]:
        """Execute concurrent load test with bounded semaphore."""
        logger.info(
            f"Starting Load Test against {self.base_url} (requests={self.total_requests}, concurrency={self.concurrency})..."
        )
        sem = asyncio.Semaphore(self.concurrency)
        results: list[dict[str, Any]] = []

        async def worker(client: httpx.AsyncClient, r_id: int):
            async with sem:
                res = await self._send_single_request(client, r_id)
                results.append(res)

        start_time = time.perf_counter()
        async with httpx.AsyncClient() as client:
            tasks = [worker(client, i) for i in range(self.total_requests)]
            await asyncio.gather(*tasks)

        total_duration_sec = time.perf_counter() - start_time
        return self._analyze_results(results, total_duration_sec)

    def _analyze_results(
        self, results: list[dict[str, Any]], total_duration_sec: float
    ) -> dict[str, Any]:
        """Compute statistical percentiles, error rate, and evaluate against SLOs."""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        latencies = sorted([r["latency_ms"] for r in successful]) if successful else [0.0]
        n_total = len(results)
        n_success = len(successful)

        p50 = statistics.median(latencies) if latencies else 0.0
        p90 = latencies[int(len(latencies) * 0.90)] if latencies else 0.0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

        throughput = n_success / max(0.001, total_duration_sec)
        error_rate = (len(failed) / n_total * 100.0) if n_total > 0 else 0.0
        availability = (n_success / n_total * 100.0) if n_total > 0 else 0.0

        # Goodput: requests succeeding within P95 SLO
        slo_compliant_reqs = [l for l in latencies if l <= SLOCriteria.MAX_P95_LATENCY_MS]
        goodput = len(slo_compliant_reqs) / max(0.001, total_duration_sec)

        # SLO evaluation
        latency_pass = p95 <= SLOCriteria.MAX_P95_LATENCY_MS
        error_pass = error_rate <= SLOCriteria.MAX_ERROR_RATE_PCT
        avail_pass = availability >= SLOCriteria.MIN_AVAILABILITY_PCT
        overall_slo_pass = latency_pass and error_pass and avail_pass

        summary = {
            "total_requests": n_total,
            "successful_requests": n_success,
            "failed_requests": len(failed),
            "concurrency": self.concurrency,
            "duration_sec": round(total_duration_sec, 2),
            "throughput_rps": round(throughput, 2),
            "goodput_rps": round(goodput, 2),
            "p50_latency_ms": round(p50, 1),
            "p90_latency_ms": round(p90, 1),
            "p95_latency_ms": round(p95, 1),
            "p99_latency_ms": round(p99, 1),
            "error_rate_pct": round(error_rate, 2),
            "availability_pct": round(availability, 2),
            "slo_eval": {
                "latency_slo_pass": latency_pass,
                "error_rate_slo_pass": error_pass,
                "availability_slo_pass": avail_pass,
                "overall_pass": overall_slo_pass,
            },
        }

        self._export_slo_report(summary)
        return summary

    def _export_slo_report(self, summary: dict[str, Any]) -> str:
        """Export comprehensive SLO Report to reports/slo_report.md."""
        report_path = self.reports_dir / "slo_report.md"

        slo = summary["slo_eval"]
        status_badge = "**PASSED (Compliant)**" if slo["overall_pass"] else "**FAILED (Violated)**"

        content = f"""# Báo cáo Kiểm định SLO và Đánh giá Tải (Phase 5)

* **Ngày đánh giá:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
* **Mục tiêu kiểm thử:** Gateway Production Control Plane (`{self.base_url}`)
* **Trạng thái tuân thủ SLO:** {status_badge}

---

## 1. Tóm tắt Kiểm định Chỉ số Cấp độ Dịch vụ (SLO Compliance)

| Tiêu chí SLO | Ngưỡng cam kết (Target) | Kết quả đo được (Actual) | Trạng thái |
|---|:---:|:---:|:---:|
| **Availability (Tính sẵn sàng)** | `>= {SLOCriteria.MIN_AVAILABILITY_PCT}%` | `{summary['availability_pct']}%` | {'✅ ĐẠT' if slo['availability_slo_pass'] else '❌ KHÔNG ĐẠT'} |
| **p95 E2E Latency** | `<= {SLOCriteria.MAX_P95_LATENCY_MS} ms` | `{summary['p95_latency_ms']} ms` | {'✅ ĐẠT' if slo['latency_slo_pass'] else '❌ KHÔNG ĐẠT'} |
| **Error Rate (Tỷ lệ lỗi)** | `<= {SLOCriteria.MAX_ERROR_RATE_PCT}%` | `{summary['error_rate_pct']}%` | {'✅ ĐẠT' if slo['error_rate_slo_pass'] else '❌ KHÔNG ĐẠT'} |

---

## 2. Chi tiết Kết quả Thử nghiệm Tải (Load Test Metrics)

* **Tổng số request gửi:** `{summary['total_requests']}`
* **Số request thành công:** `{summary['successful_requests']}`
* **Số request thất bại:** `{summary['failed_requests']}`
* **Mức độ đồng thời (Concurrency):** `{summary['concurrency']}`
* **Tổng thời gian kiểm thử:** `{summary['duration_sec']} s`
* **Throughput (Thông lượng tổng):** `{summary['throughput_rps']} req/s`
* **Goodput (Thông lượng đạt SLO):** `{summary['goodput_rps']} req/s`

### Phân bố Độ trễ (Turnaround Latency Percentiles):
* **p50 (Median):** `{summary['p50_latency_ms']} ms`
* **p90:** `{summary['p90_latency_ms']} ms`
* **p95:** `{summary['p95_latency_ms']} ms`
* **p99:** `{summary['p99_latency_ms']} ms`

---

## 3. Khuyến nghị Vận hành Production

1. **Auto-scaling Trigger:** Khi `vlm_pending_jobs > 20` hoặc `p95_latency_ms > 1200ms` kéo dài quá 2 phút, kích hoạt thêm 1 GPU Worker replica.
2. **Alerting Rules:** Cấu hình cảnh báo nếu `Error Rate > 1%` trong cửa sổ 5 phút qua Prometheus Alertmanager.
3. **Capacity Planning:** Cụm 1 GPU Tesla T4 đáp ứng tối đa 3–5 concurrent requests trước khi hàng đợi bắt đầu chịu độ trễ tích lũy.
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"SLO report successfully written to {report_path}")
        return str(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM Production Load Test and SLO Verification")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Gateway URL")
    parser.add_argument("--requests", type=int, default=50, help="Total requests to fire")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent client workers")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Output directory")
    args = parser.parse_args()

    runner = LoadTestRunner(
        base_url=args.url,
        total_requests=args.requests,
        concurrency=args.concurrency,
        reports_dir=args.reports_dir,
    )
    res = asyncio.run(runner.run_load_test())
    print("\n" + "=" * 60)
    print(f"[*] Load Test Completed: {res['successful_requests']}/{res['total_requests']} succeeded")
    print(f"[*] Throughput: {res['throughput_rps']} req/s | p95 Latency: {res['p95_latency_ms']} ms")
    print(f"[*] Overall SLO Status: {'PASSED' if res['slo_eval']['overall_pass'] else 'FAILED'}")
    print("=" * 60)
