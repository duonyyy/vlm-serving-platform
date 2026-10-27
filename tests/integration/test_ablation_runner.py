"""Integration tests for Phase 4 AblationRunner and Pareto analysis."""

import tempfile
import unittest
from pathlib import Path

from evaluation.ablation_runner import AblationRunner
from evaluation.pareto import ParetoPoint, compute_pareto_frontier


class TestAblationRunner(unittest.TestCase):
    def test_pareto_frontier_math(self) -> None:
        """Test non-dominated point identification."""
        p1 = ParetoPoint(name="P1", latency_ms=10.0, quality_score=0.9, vram_mb=5000.0, throughput_rps=100.0)
        p2 = ParetoPoint(name="P2", latency_ms=20.0, quality_score=0.95, vram_mb=6000.0, throughput_rps=50.0)
        # P3 is strictly worse than P1 (higher latency, lower quality, higher vram, lower throughput)
        p3 = ParetoPoint(name="P3", latency_ms=30.0, quality_score=0.8, vram_mb=7000.0, throughput_rps=30.0)

        frontier = compute_pareto_frontier([p1, p2, p3])
        frontier_names = [p.name for p in frontier]
        self.assertIn("P1", frontier_names)
        self.assertIn("P2", frontier_names)
        self.assertNotIn("P3", frontier_names)

    def test_ablation_runner_end_to_end(self) -> None:
        """Verify full ablation pipeline produces markdown and plot artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = AblationRunner(reports_dir=tmpdir)
            out = runner.run_full_ablation()

            self.assertTrue(Path(out["report_path"]).exists())
            self.assertTrue(Path(out["pareto_plot_path"]).exists())
            self.assertGreater(len(out["resolution"]), 0)
            self.assertGreater(out["caching"]["speedup_factor"], 1.0)
            self.assertIn("round_robin_distribution", out["scheduling"])

            # Verify markdown report content

            with open(out["report_path"], "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("# Báo cáo Nghiên cứu Tối ưu và Ablation", content)
                self.assertIn("Pareto Frontier", content)
                self.assertIn("Khuyến nghị cấu hình", content)


if __name__ == "__main__":
    unittest.main()
