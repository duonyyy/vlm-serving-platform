"""Unit tests for Prometheus Metrics Exporter."""

import unittest
from packages.telemetry.metrics import PrometheusMetrics


class TestPrometheusMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = PrometheusMetrics()

    def test_record_request_and_render(self) -> None:
        self.metrics.record_request("/v1/chat/completions", "qwen-2b", 200, 0.12)
        self.metrics.record_request("/v1/chat/completions", "qwen-2b", 200, 0.45)
        self.metrics.record_request("/v1/chat/completions", "qwen-2b", 500, 1.20)

        self.metrics.set_active_workers(3)
        self.metrics.set_pending_jobs(2)
        self.metrics.record_cache(hit=True)
        self.metrics.record_cache(hit=False)

        text = self.metrics.render_prometheus_text()
        self.assertIn("vlm_active_workers 3", text)
        self.assertIn("vlm_pending_jobs 2", text)
        self.assertIn("vlm_cache_hits_total 1", text)
        self.assertIn("vlm_cache_misses_total 1", text)
        self.assertIn('vlm_requests_total{endpoint="/v1/chat/completions",model="qwen-2b",status_code="200"} 2', text)
        self.assertIn('vlm_requests_total{endpoint="/v1/chat/completions",model="qwen-2b",status_code="500"} 1', text)
        self.assertIn("vlm_request_duration_seconds_bucket", text)
        self.assertIn("vlm_request_duration_seconds_count", text)


if __name__ == "__main__":
    unittest.main()
