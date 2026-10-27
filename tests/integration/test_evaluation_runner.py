"""Integration tests for EvaluationRunner orchestrator."""

import json
import shutil
from pathlib import Path
import unittest

from evaluation.runner import EvaluationRunner
from packages.model_adapters.mock_adapter import MockAdapter


class EvaluationRunnerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("scratch/test_eval_run")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_path = Path("evaluation/datasets/vietnamese_benchmark_50.jsonl")
        self.output_dir = self.tmp_dir / "results"
        self.reports_dir = self.tmp_dir / "reports"

    def tearDown(self) -> None:
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_evaluation_runner_end_to_end(self) -> None:
        # Create a mini dataset of 3 samples for fast integration test
        mini_manifest = self.tmp_dir / "mini_benchmark.jsonl"
        with self.dataset_path.open("r", encoding="utf-8") as in_fh, mini_manifest.open("w", encoding="utf-8") as out_fh:
            for idx, line in enumerate(in_fh):
                if idx < 3:
                    out_fh.write(line)

        adapter = MockAdapter(model_id="mock-vlm-e2e")
        runner = EvaluationRunner(
            adapter=adapter,
            dataset_path=mini_manifest,
            output_dir=self.output_dir,
            reports_dir=self.reports_dir,
            run_id="test_run_e2e",
        )
        summary = runner.run()

        self.assertEqual(summary["run_id"], "test_run_e2e")
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["performance"]["completed_requests"], 3)
        self.assertEqual(summary["performance"]["error_rate_pct"], 0.0)

        # Check output files exist
        pred_file = self.output_dir / "test_run_e2e_predictions.jsonl"
        sum_json = self.output_dir / "test_run_e2e_summary.json"
        sum_csv = self.output_dir / "test_run_e2e_summary.csv"
        report_md = self.reports_dir / "report_test_run_e2e.md"

        self.assertTrue(pred_file.is_file())
        self.assertTrue(sum_json.is_file())
        self.assertTrue(sum_csv.is_file())
        self.assertTrue(report_md.is_file())

        # Check prediction contents
        lines = pred_file.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)
        first_row = json.loads(lines[0])
        self.assertEqual(first_row["status"], "completed")
        self.assertIn("metrics", first_row)


if __name__ == "__main__":
    unittest.main()
