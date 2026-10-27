"""Unit tests for Phase 3 performance aggregation and percentiles."""

import unittest
from evaluation.performance.benchmarker import (
    aggregate_performance_results,
    calculate_percentiles,
)


class EvaluationPerformanceTests(unittest.TestCase):
    def test_calculate_percentiles(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        res = calculate_percentiles(values)
        self.assertEqual(res["min"], 10.0)
        self.assertEqual(res["max"], 100.0)
        self.assertEqual(res["mean"], 55.0)
        self.assertAlmostEqual(res["p50"], 55.0, delta=1.0)
        self.assertGreater(res["p95"], 90.0)

    def test_aggregate_performance_results(self) -> None:
        records = [
            {"status": "completed", "total_ms": 50.0, "inference_ms": 30.0, "preprocessing_ms": 5.0, "output_tokens": 20},
            {"status": "completed", "total_ms": 70.0, "inference_ms": 40.0, "preprocessing_ms": 6.0, "output_tokens": 30},
            {"status": "failed", "total_ms": 10.0, "inference_ms": 0.0, "preprocessing_ms": 1.0, "output_tokens": 0},
        ]
        profile = aggregate_performance_results(records, total_elapsed_sec=2.0)
        self.assertEqual(profile.attempted_requests, 3)
        self.assertEqual(profile.completed_requests, 2)
        self.assertEqual(profile.failed_requests, 1)
        self.assertAlmostEqual(profile.error_rate_pct, 33.33, delta=0.1)
        self.assertEqual(profile.request_throughput_rps, 1.0)
        self.assertEqual(profile.output_token_throughput_tps, 25.0)
        self.assertEqual(profile.latency_e2e_ms["min"], 50.0)
        self.assertEqual(profile.latency_e2e_ms["max"], 70.0)


if __name__ == "__main__":
    unittest.main()
