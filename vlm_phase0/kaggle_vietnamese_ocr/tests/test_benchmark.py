from __future__ import annotations

import base64
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_core import _distance, inspect_tsv, make_manifest, manifest_fingerprint, normalize, run_benchmark, score
from compare import compare, verify_pair


class BenchmarkTests(unittest.TestCase):
    def test_unicode_and_unbounded_error(self) -> None:
        self.assertEqual(normalize("Tiếng   Việt\r\nđẹp", "normalized"), "Tiếng Việt đẹp")
        self.assertNotEqual(normalize("Tiếng", "strict"), normalize("Tiếng", "strict"))
        rows = [{"ground_truth": "đ", "prediction_text": "dddd", "status": "completed"}]
        self.assertEqual(score(rows, "strict")["corpus_cer"], 4.0)
        self.assertEqual(score([{"ground_truth": "hai từ", "prediction_text": "hai", "status": "completed"}], "normalized")["corpus_whitespace_wer"], .5)
        self.assertEqual(score([{"ground_truth": "đẹp", "prediction_text": "đẹp", "status": "failed"}], "strict")["runtime_error_rate"], 1.0)

    def test_parser_keeps_duplicates_and_rejects_bad_rows(self) -> None:
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            stream = io.BytesIO()
            Image.new("RGB", (4, 3), "white").save(stream, format="PNG")
            encoded = base64.b64encode(stream.getvalue()).decode()
            tsv = root / "sample.tsv"
            with tsv.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["index", "image", "question", "answer", "extra"], delimiter="\t")
                writer.writeheader()
                writer.writerows([
                    {"index": "1", "image": encoded, "question": "OCR", "answer": "Tiếng Việt", "extra": "x"},
                    {"index": "2", "image": encoded, "question": "OCR", "answer": "Khác", "extra": "y"},
                    {"index": "3", "image": "!!!", "question": "OCR", "answer": "Hỏng", "extra": "z"},
                    {"index": "4", "image": encoded, "question": "OCR", "answer": "", "extra": "w"},
                ])
            self.assertEqual(inspect_tsv(tsv)["columns"][-1], "extra")
            rows, rejected = make_manifest(tsv, root / "out")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["duplicate_of"], "1")
            self.assertEqual(len(rejected), 2)
            self.assertEqual(len(manifest_fingerprint(rows)), 64)

    def test_comparison_blocks_partial(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            folders = [root / "a", root / "b"]
            for path, model in zip(folders, ["PaddlePaddle/PaddleOCR-VL-1.6", "deepseek-ai/DeepSeek-OCR-2"]):
                path.mkdir()
                (path / "run_manifest.json").write_text(json.dumps({"model_id": model, "dataset_revision": "r", "dataset_sha256": "s", "manifest_sha256": "m", "mode": "FULL_BENCHMARK", "valid_sample_count": 2}), encoding="utf-8")
                (path / "predictions.jsonl").write_text(json.dumps({"sample_id": "1", "image_sha256": "h", "ground_truth": "đ", "status": "completed", "prediction_text": "đ"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete benchmark"):
                verify_pair(*folders)

    def test_notebook_cells_compile(self) -> None:
        import ast
        for path in ROOT.glob("*.ipynb"):
            book = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(book["nbformat"], 4)
            for cell in book["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]))

    def test_full_comparison_exports_from_synthetic_rows(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            rows = [
                {"sample_id": "1", "image_sha256": "h1", "ground_truth": "Tiếng Việt", "status": "completed", "prediction_text": "Tiếng Việt", "latency_s": .1, "peak_device_vram_bytes": 100},
                {"sample_id": "2", "image_sha256": "h2", "ground_truth": "đẹp", "status": "completed", "prediction_text": "dep", "latency_s": .2, "peak_device_vram_bytes": 120},
            ]
            for idx, model in enumerate(["PaddlePaddle/PaddleOCR-VL-1.6", "deepseek-ai/DeepSeek-OCR-2"]):
                folder = root / str(idx)
                folder.mkdir()
                meta = {"model_id": model, "model_revision": "model-r", "dataset_revision": "r", "dataset_sha256": "s", "manifest_sha256": "m", "mode": "FULL_BENCHMARK", "valid_sample_count": 2}
                (folder / "run_manifest.json").write_text(json.dumps(meta), encoding="utf-8")
                (folder / "environment.json").write_text(json.dumps({"gpu": [{"name": "Tesla T4", "compute_capability": [7,5]}]}), encoding="utf-8")
                current = [dict(r) for r in rows]
                if idx:
                    current[1]["prediction_text"] = "đẹp"
                (folder / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in current) + "\n", encoding="utf-8")
                with (folder / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=["sample_id", "image_sha256", "ground_truth", "image_path"])
                    writer.writeheader()
                    writer.writerows({k:r[k] for k in ["sample_id", "image_sha256", "ground_truth"]} | {"image_path": ""} for r in rows)
            result = compare(root / "0", root / "1", root / "out")
            self.assertEqual(result["status"], "MEASURED")
            self.assertEqual(result["table"][0]["Better"], "DeepSeek-OCR 2")
            self.assertTrue((root / "out" / "error_analysis.html").exists())

    def test_checkpoint_resume_does_not_duplicate_samples(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            image = root / "image.png"
            image.write_bytes(b"synthetic")
            manifest = [{"sample_id": str(i), "image_sha256": f"hash{i}", "ground_truth": "abc", "image_path": str(image)} for i in range(5)]
            calls = []
            def infer(path):
                calls.append(path)
                return "abc", False
            for _ in range(2):
                rows = run_benchmark(manifest, root / "out", infer, lambda: None, lambda: (None, None, None),
                                     "SMOKE_TEST", image)
            self.assertEqual(len(rows), 5)
            self.assertEqual(len(calls), 6)  # one warm-up and five measured; rerun does nothing
            self.assertEqual(len((root / "out" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()), 5)


if __name__ == "__main__":
    unittest.main()
