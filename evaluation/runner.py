"""Comprehensive Benchmark Runner for VLM Serving (Phase 3)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.performance.benchmarker import aggregate_performance_results
from evaluation.quality.metrics import (
    compute_anls,
    compute_cer,
    compute_exact_match,
    compute_field_f1,
    compute_json_validity,
    compute_wer,
)
from evaluation.qualitative.taxonomy import ERROR_CATEGORIES, diagnose_sample_error
from packages.image_processing.decoder import decode_image
from packages.model_adapters.base import BaseModelAdapter
from packages.model_adapters.mock_adapter import MockAdapter
from packages.model_adapters.qwen_vl import QwenVLAdapter


class EvaluationRunner:
    """Orchestrates quality evaluation and performance benchmarking."""

    def __init__(
        self,
        adapter: BaseModelAdapter,
        dataset_path: Path,
        output_dir: Path,
        reports_dir: Path,
        run_id: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        self.run_id = run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self) -> list[dict[str, Any]]:
        """Load and validate benchmark samples."""
        samples: list[dict[str, Any]] = []
        with self.dataset_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def run(self) -> dict[str, Any]:
        """Execute full evaluation run."""
        samples = self.load_dataset()
        print(f"Starting Benchmark [{self.run_id}] with {len(samples)} samples on model [{self.adapter.model_id}]...")

        if self.adapter.status != "READY":
            print(f"Loading adapter [{self.adapter.model_id}]...")
            self.adapter.load()

        records: list[dict[str, Any]] = []
        overall_start = time.perf_counter()

        for idx, sample in enumerate(samples, 1):
            sample_id = sample["sample_id"]
            task_type = sample["task_type"]
            prompt = sample["prompt"]
            gold = sample["ground_truth"]
            image_rel = sample["image_path"]

            # Resolve image path relative to dataset directory or evaluation directory
            image_path = Path(image_rel)
            if not image_path.is_file():
                candidates = [
                    self.dataset_path.parent / image_rel,
                    Path(__file__).resolve().parent / "datasets" / image_rel,
                    Path(__file__).resolve().parent.parent / "evaluation" / "datasets" / image_rel,
                    Path(__file__).resolve().parent.parent / image_rel,
                ]
                for cand in candidates:
                    if cand.is_file():
                        image_path = cand
                        break

            # Measure end-to-end sample time
            sample_start = time.perf_counter()
            status = "completed"
            prediction = ""
            error_message = None
            prep_ms = 0.0
            infer_ms = 0.0
            in_tokens = 0
            out_tokens = 0

            try:
                # 1. Decode image
                decoded_img, _ = decode_image(image_path)

                # 2. Infer
                res = self.adapter.infer(prompt=prompt, image=decoded_img, max_tokens=512)
                prediction = res.prediction_text
                prep_ms = res.preprocessing_ms
                infer_ms = res.inference_ms
                in_tokens = res.input_tokens
                out_tokens = res.output_tokens

            except Exception as exc:
                status = "failed"
                error_message = str(exc)

            sample_total_ms = round((time.perf_counter() - sample_start) * 1000, 3)

            # Compute quality metrics based on task type
            metrics: dict[str, Any] = {}
            error_tags: list[str] = []

            if status == "completed":
                if task_type == "invoice":
                    # Check JSON validity and field F1
                    valid_json, parsed_dict, err = compute_json_validity(prediction)
                    metrics["json_valid"] = valid_json
                    if valid_json and isinstance(gold, dict) and isinstance(parsed_dict, dict):
                        f1_res = compute_field_f1(gold, parsed_dict)
                        metrics["f1"] = f1_res["f1"]
                        metrics["precision"] = f1_res["precision"]
                        metrics["recall"] = f1_res["recall"]
                        metrics["field_matches"] = f1_res["fields"]
                    else:
                        metrics["f1"] = 0.0
                        error_tags.append("SCHEMA_FORMAT_ERROR")

                elif task_type == "ocr":
                    ref_text = str(gold)
                    cer = compute_cer(ref_text, prediction, mode="normalized")
                    wer = compute_wer(ref_text, prediction, mode="normalized")
                    em = compute_exact_match(ref_text, prediction, mode="normalized")
                    metrics["cer"] = cer
                    metrics["wer"] = wer
                    metrics["exact_match"] = em

                    if cer > 0.05:
                        error_tags.extend(diagnose_sample_error(ref_text, prediction, status=status, task_type=task_type))

                elif task_type == "vqa":
                    ref_text = str(gold)
                    anls = compute_anls(ref_text, prediction)
                    em = compute_exact_match(ref_text, prediction, mode="normalized")
                    metrics["anls"] = anls
                    metrics["exact_match"] = em

                    if anls < 0.8:
                        error_tags.extend(diagnose_sample_error(ref_text, prediction, status=status, task_type=task_type))

            else:
                error_tags.append("SCHEMA_FORMAT_ERROR")

            record = {
                "sample_id": sample_id,
                "task_type": task_type,
                "category": sample.get("category"),
                "status": status,
                "error_message": error_message,
                "prediction": prediction,
                "ground_truth": gold,
                "metrics": metrics,
                "error_tags": error_tags,
                "preprocessing_ms": prep_ms,
                "inference_ms": infer_ms,
                "total_ms": sample_total_ms,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
            }
            records.append(record)

            if idx % 10 == 0 or idx == len(samples):
                print(f"  Processed [{idx}/{len(samples)}] samples...")

        total_elapsed_sec = time.perf_counter() - overall_start

        # Aggregate performance metrics
        perf_profile = aggregate_performance_results(records, total_elapsed_sec)

        # Aggregate quality metrics by task type
        quality_summary = self._aggregate_quality(records)

        # Aggregate error taxonomy
        all_tags = [tag for r in records for tag in r.get("error_tags", [])]
        taxonomy_counts = dict(Counter(all_tags))

        summary = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_id": self.adapter.model_id,
            "sample_count": len(samples),
            "performance": perf_profile.model_dump(),
            "quality": quality_summary,
            "error_taxonomy": taxonomy_counts,
        }

        # Export outputs
        self._export_results(records, summary)
        self._generate_report(summary, records)
        self._generate_plots(perf_profile, quality_summary)

        print(f"Benchmark [{self.run_id}] completed in {total_elapsed_sec:.2f}s!")
        return summary

    def _aggregate_quality(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute average scores across tasks."""
        invoice_records = [r for r in records if r["task_type"] == "invoice" and r["status"] == "completed"]
        ocr_records = [r for r in records if r["task_type"] == "ocr" and r["status"] == "completed"]
        vqa_records = [r for r in records if r["task_type"] == "vqa" and r["status"] == "completed"]

        # Invoices
        invoice_f1s = [r["metrics"].get("f1", 0.0) for r in invoice_records]
        json_valids = [1 for r in invoice_records if r["metrics"].get("json_valid")]
        mean_inv_f1 = round(sum(invoice_f1s) / len(invoice_f1s), 4) if invoice_f1s else 0.0
        json_valid_rate = round(len(json_valids) / len(invoice_records) * 100, 2) if invoice_records else 0.0

        # OCR
        cers = [r["metrics"].get("cer", 1.0) for r in ocr_records]
        wers = [r["metrics"].get("wer", 1.0) for r in ocr_records]
        ocr_ems = [1 for r in ocr_records if r["metrics"].get("exact_match")]
        mean_cer = round(sum(cers) / len(cers), 4) if cers else 0.0
        mean_wer = round(sum(wers) / len(wers), 4) if wers else 0.0
        ocr_em_rate = round(len(ocr_ems) / len(ocr_records) * 100, 2) if ocr_records else 0.0

        # VQA
        anls_scores = [r["metrics"].get("anls", 0.0) for r in vqa_records]
        vqa_ems = [1 for r in vqa_records if r["metrics"].get("exact_match")]
        mean_anls = round(sum(anls_scores) / len(anls_scores), 4) if anls_scores else 0.0
        vqa_em_rate = round(len(vqa_ems) / len(vqa_records) * 100, 2) if vqa_records else 0.0

        return {
            "invoice": {
                "count": len(invoice_records),
                "mean_f1": mean_inv_f1,
                "json_validity_rate_pct": json_valid_rate,
            },
            "ocr": {
                "count": len(ocr_records),
                "mean_cer": mean_cer,
                "mean_wer": mean_wer,
                "exact_match_pct": ocr_em_rate,
            },
            "vqa": {
                "count": len(vqa_records),
                "mean_anls": mean_anls,
                "exact_match_pct": vqa_em_rate,
            },
        }

    def _export_results(self, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
        """Export predictions JSONL, summary JSON, and summary CSV."""
        # 1. Predictions JSONL
        jsonl_path = self.output_dir / f"{self.run_id}_predictions.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 2. Summary JSON
        summary_path = self.output_dir / f"{self.run_id}_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

        # 3. Summary CSV
        csv_path = self.output_dir / f"{self.run_id}_summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["sample_id", "task_type", "category", "status", "total_ms", "inference_ms", "output_tokens", "error_tags"])
            for r in records:
                writer.writerow([
                    r["sample_id"],
                    r["task_type"],
                    r.get("category", ""),
                    r["status"],
                    r["total_ms"],
                    r["inference_ms"],
                    r["output_tokens"],
                    ";".join(r.get("error_tags", [])),
                ])

    def _generate_report(self, summary: dict[str, Any], records: list[dict[str, Any]]) -> Path:
        """Create a GitHub-flavored Markdown evaluation report."""
        report_path = self.reports_dir / f"report_{self.run_id}.md"
        perf = summary["performance"]
        qual = summary["quality"]
        tax = summary["error_taxonomy"]

        md = f"""# Báo cáo Đánh giá Benchmark VLM (Phase 3)

* **Run ID:** `{self.run_id}`
* **Thời gian thực hiện:** `{summary['timestamp']}`
* **Mô hình kiểm thử:** `{summary['model_id']}`
* **Số lượng mẫu:** `{summary['sample_count']}`

---

## 1. Tóm tắt chỉ số Serving Performance

| Chỉ số | Giá trị | Đơn vị |
|---|---:|---|
| **Tổng số request** | {perf['attempted_requests']} | mẫu |
| **Tỷ lệ lỗi (Error Rate)** | {perf['error_rate_pct']}% | % |
| **Request Throughput** | {perf['request_throughput_rps']} | req/s |
| **Output Token Throughput** | {perf['output_token_throughput_tps']} | tokens/s |
| **p50 E2E Latency** | {perf['latency_e2e_ms']['p50']} | ms |
| **p95 E2E Latency** | {perf['latency_e2e_ms']['p95']} | ms |
| **p99 E2E Latency** | {perf['latency_e2e_ms']['p99']} | ms |
| **p50 Inference Latency** | {perf['latency_inference_ms']['p50']} | ms |
| **p95 Inference Latency** | {perf['latency_inference_ms']['p95']} | ms |

---

## 2. Tóm tắt chỉ số chất lượng (Quality Benchmark)

### A. Trích xuất hóa đơn tiếng Việt (Invoice Extraction - 20 mẫu)
* **Mean Field F1:** `{qual['invoice']['mean_f1']:.4f}`
* **JSON Validity Rate:** `{qual['invoice']['json_validity_rate_pct']}%`

### B. Nhận dạng ký tự quang học (Document OCR - 15 mẫu)
* **Mean CER (Character Error Rate):** `{qual['ocr']['mean_cer']:.4f}`
* **Mean WER (Word Error Rate):** `{qual['ocr']['mean_wer']:.4f}`
* **Tỷ lệ khớp tuyệt đối (Exact Match):** `{qual['ocr']['exact_match_pct']}%`

### C. Hỏi đáp thị giác (Multimodal VQA - 15 mẫu)
* **Mean ANLS:** `{qual['vqa']['mean_anls']:.4f}`
* **Tỷ lệ khớp tuyệt đối (Exact Match):** `{qual['vqa']['exact_match_pct']}%`

---

## 3. Phân loại lỗi định tính (Error Taxonomy)

Theo phân loại lỗi tại Mục 8.3 Kế hoạch triển khai:

| Mã lỗi | Mô tả | Số ca xuất hiện | Tỷ lệ (%) |
|---|---|---:|---:|
"""
        total_errors = sum(tax.values())
        for code, desc in ERROR_CATEGORIES.items():
            count = tax.get(code, 0)
            pct = round((count / total_errors) * 100, 1) if total_errors > 0 else 0.0
            md += f"| `{code}` | {desc} | {count} | {pct}% |\n"

        md += f"""
---

## 4. Danh sách các mẫu lỗi tiêu biểu
"""
        failed_or_tagged = [r for r in records if r.get("error_tags")]
        if not failed_or_tagged:
            md += "\n*Không có ca lỗi nghiêm trọng nào được ghi nhận.*\n"
        else:
            md += "\n| Sample ID | Nhiệm vụ | Lỗi phát hiện | Dự đoán tóm tắt |\n|---|---|---|---|\n"
            for r in failed_or_tagged[:10]:
                pred_short = r['prediction'][:60].replace("\n", " ") + "..." if len(r['prediction']) > 60 else r['prediction'].replace("\n", " ")
                md += f"| `{r['sample_id']}` | {r['task_type']} | {', '.join(r.get('error_tags', []))} | {pred_short} |\n"

        report_path.write_text(md, encoding="utf-8")
        return report_path

    def _generate_plots(self, perf: Any, qual: dict[str, Any]) -> None:
        """Generate graphical visualization of the benchmark results."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Latency Bar Plot
            latencies = [
                perf.latency_e2e_ms.get("p50", 0),
                perf.latency_e2e_ms.get("p90", 0),
                perf.latency_e2e_ms.get("p95", 0),
                perf.latency_e2e_ms.get("p99", 0),
            ]
            labels = ["p50", "p90", "p95", "p99"]
            ax1.bar(labels, latencies, color="#3B82F6")
            ax1.set_title("E2E Latency Percentiles (ms)")
            ax1.set_ylabel("Milliseconds")
            for i, v in enumerate(latencies):
                ax1.text(i, v + max(latencies) * 0.02, f"{v:.1f}", ha="center", fontsize=9)

            # Quality Metrics Bar Plot
            quality_labels = ["Invoice F1", "JSON Valid %", "VQA ANLS", "OCR 1-CER"]
            quality_vals = [
                qual["invoice"]["mean_f1"],
                qual["invoice"]["json_validity_rate_pct"] / 100.0,
                qual["vqa"]["mean_anls"],
                max(0.0, 1.0 - qual["ocr"]["mean_cer"]),
            ]
            ax2.bar(quality_labels, quality_vals, color="#10B981")
            ax2.set_title("Quality Scores (Normalized 0.0 - 1.0)")
            ax2.set_ylim(0, 1.15)
            for i, v in enumerate(quality_vals):
                ax2.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)

            plt.tight_layout()
            plot_file = self.reports_dir / f"{self.run_id}_plot.png"
            plt.savefig(plot_file, dpi=150)
            plt.close()
        except Exception as exc:
            print(f"Plot generation skipped: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 VLM Benchmark")
    parser.add_argument("--dataset", type=str, default="evaluation/datasets/vietnamese_benchmark_50.jsonl")
    parser.add_argument("--adapter", type=str, choices=["mock", "qwen"], default="mock")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="evaluation/results")
    parser.add_argument("--reports-dir", type=str, default="evaluation/reports")
    args = parser.parse_args()

    if args.adapter == "mock":
        adapter = MockAdapter(model_id=args.model_id or "mock-vlm-2b")
    else:
        adapter = QwenVLAdapter(model_id=args.model_id or "Qwen/Qwen3-VL-2B-Instruct")
        if args.device:
            adapter.device_name = args.device

    runner = EvaluationRunner(
        adapter=adapter,
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        reports_dir=Path(args.reports_dir),
    )
    runner.run()


if __name__ == "__main__":
    main()
