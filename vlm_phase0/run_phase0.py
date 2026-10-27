"""Run the 20-image Qwen3-VL feasibility smoke on Kaggle T4."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .core import MODEL_ID, hardware_report, require_t4, sha256, utc_now, validate_manifest, write_json


def execute(cases: list[dict], baseline: object, output_dir: Path, manifest_sha: str) -> dict:
    """Inference success is mechanical; visual correctness requires human review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "smoke_results.jsonl"
    review_path = output_dir / "smoke_review.csv"
    results = []
    with results_path.open("w", encoding="utf-8") as stream, review_path.open("w", encoding="utf-8", newline="") as reviews:
        writer = csv.DictWriter(reviews, fieldnames=["sample_id", "category", "expected_observation", "prediction_text", "verdict", "reviewer_note"])
        writer.writeheader()
        for case in cases:
            row = {"sample_id": case["sample_id"], "category": case["category"],
                   "image_sha256": case["image_sha256"], "image_bytes": case["image_bytes"],
                   "width": case["width"], "height": case["height"], "model_id": MODEL_ID,
                   "model_revision": baseline.revision, "manifest_sha256": manifest_sha,
                   "resize_longest_edge": getattr(baseline, "resize_longest_edge", None),
                   "captured_at_utc": utc_now(), "status": "failed"}
            start = time.perf_counter()
            try:
                row.update(baseline.infer(Path(case["image_path"]), case["prompt"]))
                row["status"] = "completed" if row["prediction_text"] else "empty_output"
            except Exception as exc:
                row["status"] = "oom" if "out of memory" in str(exc).lower() else "failed"
                row["error_type"] = type(exc).__name__
                row["error_message"] = str(exc)[:500]
            row["total_ms"] = round((time.perf_counter() - start) * 1000, 3)
            results.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            writer.writerow({"sample_id": case["sample_id"], "category": case["category"],
                             "expected_observation": case["expected_observation"],
                             "prediction_text": row.get("prediction_text", ""),
                             "verdict": "UNREVIEWED", "reviewer_note": ""})
            reviews.flush()
    successes = sum(row["status"] == "completed" for row in results)
    report = {"status": "MEASURED", "model_id": MODEL_ID, "model_revision": baseline.revision,
              "manifest_sha256": manifest_sha, "attempted": len(results), "valid_nonempty_outputs": successes,
              "model_config": {"dtype": "float16", "attention": "eager", "device": "cuda:0",
                               "batch_size": 1, "max_new_tokens": 128, "do_sample": False,
                               "resize_longest_edge": getattr(baseline, "resize_longest_edge", None),
                               "resize_filter": "LANCZOS" if getattr(baseline, "resize_longest_edge", None) else None},
              "runtime_exit_met": successes >= 20, "visual_correctness": "UNREVIEWED",
              "note": "Nonempty output does not prove the answer is grounded in the image. Review smoke_review.csv."}
    write_json(output_dir / "smoke_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-revision", default=None, help="Optional 40-character Hugging Face commit SHA")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    hardware = hardware_report()
    write_json(args.output_dir / "hardware_report.json", hardware)
    if args.preflight_only:
        return 0
    if not args.manifest or not args.dataset_root:
        parser.error("--manifest and --dataset-root are required for inference")
    require_t4(hardware)
    cases = validate_manifest(args.manifest, args.dataset_root)
    from .qwen import QwenBaseline
    try:
        baseline = QwenBaseline(args.model_revision)
    except Exception as exc:
        write_json(args.output_dir / "smoke_report.json", {
            "status": "MODEL_LOAD_FAILED", "model_id": MODEL_ID,
            "manifest_sha256": sha256(args.manifest),
            "error_type": type(exc).__name__, "error_message": str(exc)[:500],
            "captured_at_utc": utc_now(),
        })
        raise
    report = execute(cases, baseline, args.output_dir, sha256(args.manifest))
    return 0 if report["runtime_exit_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
