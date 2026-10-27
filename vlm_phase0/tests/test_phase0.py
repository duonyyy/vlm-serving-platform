from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vlm_phase0.core import validate_manifest
from vlm_phase0.qwen import MAX_IMAGE_EDGE, prepare_image
from vlm_phase0.run_phase0 import execute


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parents[1])
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        self.manifest = self.root / "manifest.jsonl"
        rows = []
        categories = ["scene", "document", "text", "chart"]
        for i in range(20):
            path = self.root / "images" / f"{i}.png"
            Image.new("RGB", (8, 8), color=(i, 10, 20)).save(path)
            rows.append({"sample_id": f"s{i:02d}", "image_path": f"images/{i}.png",
                         "category": categories[i % 4], "prompt": "What is visible?",
                         "expected_observation": f"Synthetic color {i}"})
        self.rows = rows
        self.write()

    def write(self) -> None:
        self.manifest.write_text("\n".join(json.dumps(row) for row in self.rows) + "\n", encoding="utf-8")

    def test_valid_manifest_has_hashes_and_four_strata(self) -> None:
        cases = validate_manifest(self.manifest, self.root)
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({case["image_sha256"] for case in cases}), 20)

    def test_duplicate_bytes_fail_closed(self) -> None:
        (self.root / "images" / "19.png").write_bytes((self.root / "images" / "18.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "duplicate image bytes"):
            validate_manifest(self.manifest, self.root)

    def test_traversal_fails_closed(self) -> None:
        self.rows[0]["image_path"] = "../other.png"
        self.write()
        with self.assertRaisesRegex(ValueError, "relative to dataset root"):
            validate_manifest(self.manifest, self.root)


class ImagePreprocessingTests(unittest.TestCase):
    def test_large_image_is_bounded_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as temp_dir:
            path = Path(temp_dir) / "large.png"
            Image.new("RGB", (1920, 1280), color=(12, 34, 56)).save(path)
            source_bytes = path.read_bytes()
            image, original_size = prepare_image(path)
            self.assertEqual(original_size, (1920, 1280))
            self.assertEqual(image.size, (1280, 853))
            self.assertEqual(max(image.size), MAX_IMAGE_EDGE)
            self.assertEqual(path.read_bytes(), source_bytes)


class RunnerTests(ManifestTests):
    def test_output_remains_unreviewed_and_failure_is_visible(self) -> None:
        cases = validate_manifest(self.manifest, self.root)

        class FakeBaseline:
            revision = "a" * 40
            resize_longest_edge = MAX_IMAGE_EDGE

            def infer(self, image_path: Path, prompt: str) -> dict:
                if image_path.name == "19.png":
                    raise RuntimeError("CUDA out of memory")
                return {"prediction_text": "synthetic answer", "inference_ms": 1.0}

        output = self.root / "out"
        report = execute(cases, FakeBaseline(), output, "b" * 64)
        self.assertEqual(report["valid_nonempty_outputs"], 19)
        self.assertFalse(report["runtime_exit_met"])
        self.assertEqual(report["visual_correctness"], "UNREVIEWED")
        self.assertEqual(report["model_config"]["resize_longest_edge"], MAX_IMAGE_EDGE)
        rows = [json.loads(line) for line in (output / "smoke_results.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["status"], "oom")
        self.assertTrue(all(row["resize_longest_edge"] == MAX_IMAGE_EDGE for row in rows))
        with (output / "smoke_review.csv").open(encoding="utf-8", newline="") as stream:
            reviews = list(csv.DictReader(stream))
        self.assertEqual(len(reviews), 20)
        self.assertTrue(all(row["verdict"] == "UNREVIEWED" for row in reviews))


if __name__ == "__main__":
    unittest.main()
