"""CPU-only comparison. Refuses to produce a full result from partial or mismatched runs."""
from __future__ import annotations

import csv
import difflib
import html
import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any

from benchmark_core import _distance, normalize, read_jsonl, score, write_json


def _read_run(folder: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    return config, read_jsonl(folder / "predictions.jsonl")


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["sample_id"]), str(row["image_sha256"])


def verify_pair(pa: Path, pb: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    a_meta, a = _read_run(pa)
    b_meta, b = _read_run(pb)
    if a_meta["model_id"] != "PaddlePaddle/PaddleOCR-VL-1.6" or b_meta["model_id"] != "deepseek-ai/DeepSeek-OCR-2":
        raise ValueError("Wrong model IDs")
    for field in ("dataset_revision", "dataset_sha256", "manifest_sha256", "mode"):
        if a_meta[field] != b_meta[field]:
            raise ValueError(f"Mismatch in {field}")
    if a_meta["mode"] != "FULL_BENCHMARK":
        raise ValueError("Comparison requires FULL_BENCHMARK outputs")
    aa, bb = {_key(x): x for x in a}, {_key(x): x for x in b}
    if len(aa) != len(a) or len(bb) != len(b):
        raise ValueError("Duplicate checkpoint row")
    if aa.keys() != bb.keys():
        missing_a = sorted(bb.keys() - aa.keys())
        missing_b = sorted(aa.keys() - bb.keys())
        raise ValueError(f"Incomplete common set: missing Paddle={missing_a[:10]}, DeepSeek={missing_b[:10]}")
    expected = a_meta["valid_sample_count"]
    if len(aa) != expected:
        raise ValueError(f"Incomplete benchmark: {len(aa)} / {expected}")
    with (pa / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        manifest = {_key(r): r for r in csv.DictReader(fh)}
    if len(manifest) != expected or aa.keys() != manifest.keys():
        raise ValueError("Prediction keys do not match full manifest")
    for key in aa:
        if aa[key]["ground_truth"] != bb[key]["ground_truth"]:
            raise ValueError(f"Ground truth mismatch for {key}")
        if aa[key]["ground_truth"] != manifest[key]["ground_truth"]:
            raise ValueError(f"Ground truth differs from manifest for {key}")
        if aa[key]["status"] not in {"completed", "failed", "oom", "timeout"} or bb[key]["status"] not in {"completed", "failed", "oom", "timeout"}:
            raise ValueError(f"Unknown status for {key}")
    for folder in (pa, pb):
        env = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
        gpu = env.get("gpu", [{}])[0]
        if "T4" not in gpu.get("name", "") or gpu.get("compute_capability") != [7, 5]:
            raise ValueError(f"Non-T4 or unverified GPU environment: {folder}")
    keys = sorted(aa)
    return [aa[k] for k in keys], [bb[k] for k in keys], a_meta


def category_flags(gt: str, pred: str) -> dict[str, bool]:
    """Heuristic tags from actual edits; not mutually exclusive or causal labels."""
    g, p = normalize(gt, "normalized"), normalize(pred, "normalized")
    strip_marks = lambda s: "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return {
        "diacritic_only": g != p and strip_marks(g).casefold() == strip_marks(p).casefold(),
        "d_vs_stroke": g != p and g.replace("đ", "d").replace("Đ", "D") == p.replace("đ", "d").replace("Đ", "D"),
        "case_only": g != p and g.casefold() == p.casefold(),
        "digit_difference": re.sub(r"\D", "", g) != re.sub(r"\D", "", p),
        "punctuation_difference": "".join(c for c in g if unicodedata.category(c).startswith("P")) != "".join(c for c in p if unicodedata.category(c).startswith("P")),
        "possible_omission": len(p) < len(g) and len(p) < 0.8 * len(g),
        "possible_addition_or_repetition": len(p) > 1.2 * len(g),
    }


def _cer(gt: str, pred: str) -> float:
    return _distance(gt, pred) / max(1, len(gt))


def _metric_value(rows: list[dict[str, Any]], name: str) -> float | None:
    import numpy as np
    good = [r for r in rows if r["status"] == "completed" and r.get("latency_s") is not None]
    if name == "throughput":
        total = sum(float(r["latency_s"]) for r in rows if r.get("latency_s") is not None)
        return len(good) / total if total else None
    if name.startswith("latency"):
        q = 50 if name == "latency_p50" else 95
        return float(np.percentile([float(r["latency_s"]) for r in good], q)) if good else None
    if name == "peak_vram":
        vals = [r.get("peak_device_vram_bytes") or r.get("peak_reserved_bytes") for r in rows]
        return max((int(v) for v in vals if v is not None), default=None)
    return None


def _plot(a: list[dict[str, Any]], b: list[dict[str, Any]], out: Path, sa: dict[str, Any], sb: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use("default")
    names = ["PaddleOCR-VL 1.6", "DeepSeek-OCR 2"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    ax.bar(x - .18, [sa["corpus_cer"], sa["corpus_whitespace_wer"], sa["exact_match_rate"]], .36, label=names[0])
    ax.bar(x + .18, [sb["corpus_cer"], sb["corpus_whitespace_wer"], sb["exact_match_rate"]], .36, label=names[1])
    ax.set_xticks(x, ["CER", "WER", "Exact match"]); ax.legend(); ax.set_ylabel("Rate")
    fig.tight_layout(); fig.savefig(out / "quality.png", dpi=180); plt.close(fig)
    for title, values, filename in [
        ("Per-image CER", [[_cer(normalize(r["ground_truth"], "normalized"), normalize(r.get("prediction_text", "") if r["status"] == "completed" else "", "normalized")) for r in rows] for rows in (a,b)], "cer_boxplot.png"),
        ("Latency", [[float(r["latency_s"]) for r in rows if r["status"] == "completed"] for rows in (a,b)], "latency_histogram.png")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        if filename.startswith("cer"):
            ax.boxplot(values, tick_labels=names, showfliers=True)
        else:
            for v, label in zip(values, names):
                if v: ax.hist(v, bins=min(20, max(3, len(v)//3)), alpha=.55, label=label)
            ax.legend(); ax.set_xlabel("Seconds")
        ax.set_title(title); fig.tight_layout(); fig.savefig(out / filename, dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for rows, label in zip((a,b), names):
        xs = [float(r["latency_s"]) for r in rows if r["status"] == "completed"]
        ys = [_cer(normalize(r["ground_truth"], "normalized"), normalize(r.get("prediction_text", ""), "normalized")) for r in rows if r["status"] == "completed"]
        ax.scatter(xs, ys, label=label, alpha=.65)
    ax.set(xlabel="Latency (s)", ylabel="CER"); ax.legend(); fig.tight_layout(); fig.savefig(out / "cer_vs_latency.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    vram_values = [_metric_value(rows, "peak_vram") for rows in (a,b)]
    ax.bar(names, [(v or 0) / 2**30 for v in vram_values])
    for i, v in enumerate(vram_values):
        if v is None: ax.text(i, 0, "NOT_EVALUATED", ha="center", va="bottom")
    ax.set_ylabel("Peak reported VRAM (GiB)"); fig.tight_layout(); fig.savefig(out / "vram.png", dpi=180); plt.close(fig)
    wins = [0,0,0]
    for ra, rb in zip(a,b):
        ca = _cer(normalize(ra["ground_truth"], "normalized"), normalize(ra.get("prediction_text", "") if ra["status"] == "completed" else "", "normalized"))
        cb = _cer(normalize(rb["ground_truth"], "normalized"), normalize(rb.get("prediction_text", "") if rb["status"] == "completed" else "", "normalized"))
        wins[0 if ca < cb else 1 if cb < ca else 2] += 1
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(["Paddle wins", "DeepSeek wins", "Tie"], wins)
    ax.set_ylabel("Images"); fig.tight_layout(); fig.savefig(out / "wins.png", dpi=180); plt.close(fig)


def compare(paddle_dir: Path, deepseek_dir: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    a, b, meta = verify_pair(paddle_dir, deepseek_dir)
    deepseek_meta = json.loads((deepseek_dir / "run_manifest.json").read_text(encoding="utf-8"))
    env_a = json.loads((paddle_dir / "environment.json").read_text(encoding="utf-8"))
    env_b = json.loads((deepseek_dir / "environment.json").read_text(encoding="utf-8"))
    strict_a, strict_b = score(a, "strict"), score(b, "strict")
    sa, sb = score(a, "normalized"), score(b, "normalized")
    metrics = [
        ("CER", sa["corpus_cer"], sb["corpus_cer"], False),
        ("WER (whitespace)", sa["corpus_whitespace_wer"], sb["corpus_whitespace_wer"], False),
        ("Exact Match", sa["exact_match_rate"], sb["exact_match_rate"], True),
        ("Latency p50", _metric_value(a,"latency_p50"), _metric_value(b,"latency_p50"), False),
        ("Latency p95", _metric_value(a,"latency_p95"), _metric_value(b,"latency_p95"), False),
        ("Throughput", _metric_value(a,"throughput"), _metric_value(b,"throughput"), True),
        ("Peak VRAM", _metric_value(a,"peak_vram"), _metric_value(b,"peak_vram"), False),
    ]
    table = []
    for name, x, y, higher in metrics:
        better = "NOT_EVALUATED" if x is None or y is None else "Tie" if x == y else ("PaddleOCR-VL 1.6" if (x > y) == higher else "DeepSeek-OCR 2")
        table.append({"Metric": name, "PaddleOCR-VL 1.6": x, "DeepSeek-OCR 2": y, "Better": better})
    summary = {"status": "MEASURED", "dataset_revision": meta["dataset_revision"], "n": len(a),
               "strict": {"paddle": strict_a, "deepseek": strict_b}, "normalized": {"paddle": sa, "deepseek": sb}, "table": table}
    combined = []
    for ra, rb in zip(a,b):
        gt = ra["ground_truth"]
        ca = _cer(normalize(gt,"normalized"), normalize(ra.get("prediction_text","") if ra["status"] == "completed" else "","normalized"))
        cb = _cer(normalize(gt,"normalized"), normalize(rb.get("prediction_text","") if rb["status"] == "completed" else "","normalized"))
        combined.append({"sample_id": ra["sample_id"], "image_sha256": ra["image_sha256"], "ground_truth": gt,
                         "paddle_prediction": ra.get("prediction_text", ""), "deepseek_prediction": rb.get("prediction_text", ""),
                         "paddle_status": ra["status"], "deepseek_status": rb["status"], "paddle_cer": ca, "deepseek_cer": cb,
                         "paddle_flags": json.dumps(category_flags(gt,ra.get("prediction_text","")),ensure_ascii=False),
                         "deepseek_flags": json.dumps(category_flags(gt,rb.get("prediction_text","")),ensure_ascii=False)})
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(combined[0])); writer.writeheader(); writer.writerows(combined)
    _plot(a,b,out,sa,sb)
    categories = {model: {kind: sum(category_flags(r["ground_truth"], r.get("prediction_text", "") if r["status"] == "completed" else "")[kind] for r in rows)
                          for kind in category_flags("", "")}
                  for model, rows in (("PaddleOCR-VL 1.6", a), ("DeepSeek-OCR 2", b))}
    summary["error_categories_heuristic"] = categories
    write_json(out / "comparison_summary.json", summary)
    report = ["# Báo cáo OCR tiếng Việt", "",
              "## 1. Dataset và giới hạn nghiên cứu", "",
              f"CC-OCR Vietnamese_150.tsv, commit `{meta['dataset_revision']}`, SHA256 TSV `{meta['dataset_sha256']}`, {len(a)} ảnh hợp lệ. Chỉ tập này được chấm; số mẫu hợp lệ có thể khác 150. Nhãn `answer` không được kiểm định độc lập ngoài cổng xác nhận trong notebook.", "",
              "## 2. Mô hình", "",
              f"PaddleOCR-VL 1.6: commit tham chiếu `{meta['model_revision']}`, pipeline chính thức gồm layout + VLM. DeepSeek-OCR 2: commit `{deepseek_meta['model_revision']}`, prompt Free OCR, FP16/eager trên T4. Bản vá: `{deepseek_meta.get('patch')}`.", "",
              "## 3. Phần cứng và môi trường", "",
              f"Paddle: {env_a['gpu'][0]['name']}, Python {env_a.get('python')}, Torch {env_a.get('torch')}, CUDA {env_a.get('torch_cuda_runtime')}, driver {env_a.get('nvidia_smi')}. DeepSeek: {env_b['gpu'][0]['name']}, Python {env_b.get('python')}, Torch {env_b.get('torch')}, CUDA {env_b.get('torch_cuda_runtime')}, driver {env_b.get('nvidia_smi')}.", "",
              "## 4. Phương pháp thực nghiệm", "",
              "Ảnh, nhãn và sample ID/hash được kiểm tra chung. Batch 1, cuda:0, seed 42, warm-up riêng. Strict giữ nguyên ký tự; normalized chỉ NFC và khoảng trắng. CER/WER dùng edit distance; WER token theo khoảng trắng. Lỗi runtime/OOM/timeout được tính như dự đoán rỗng trong chất lượng đầu cuối và báo riêng.", "",
              "## 5. Bảng kết quả normalized", "",
              "| Metric | PaddleOCR-VL 1.6 | DeepSeek-OCR 2 | Better |", "|---|---:|---:|---|"]
    for row in table:
        fmt = lambda x: "NOT_EVALUATED" if x is None else f"{x:.4f}"
        report.append(f"| {row['Metric']} | {fmt(row['PaddleOCR-VL 1.6'])} | {fmt(row['DeepSeek-OCR 2'])} | {row['Better']} |")
    report += ["", f"Strict CER: Paddle {strict_a['corpus_cer']:.4f}, DeepSeek {strict_b['corpus_cer']:.4f}; strict whitespace WER: Paddle {strict_a['corpus_whitespace_wer']:.4f}, DeepSeek {strict_b['corpus_whitespace_wer']:.4f}.",
               "", "## 6. Phân tích lỗi tiếng Việt", "",
               "Các đếm dưới đây là heuristic từ chuỗi khác biệt, có thể chồng lấn và không chứng minh nguyên nhân lỗi."]
    for model, counts in categories.items():
        report.append(f"- {model}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    report += ["", "Xem `error_analysis.html` để đối chiếu ảnh, nhãn, hai dự đoán và diff của tối đa 10 ảnh lỗi theo quy tắc chọn cố định.",
               "", "## 7. Độ chính xác và chi phí tài nguyên", "",
               f"Runtime error rate: Paddle {sa['runtime_error_rate']:.4f}, DeepSeek {sb['runtime_error_rate']:.4f}. Latency và VRAM nằm trong bảng; NVML đo toàn GPU mỗi 50 ms nên đây là ước lượng có giới hạn. Throughput là số ảnh thành công chia tổng thời gian inference của mọi ảnh đã thử.",
               "", "## 8. Threats to validity", "",
               "Tập chỉ có một ngôn ngữ và tối đa 150 ảnh. Nhãn chưa có kiểm định độc lập. FP16 DeepSeek là bản port chưa được tác giả xác nhận. Paddle gồm layout analysis; so sánh này không tách vision encoder. Markdown-to-text có thể ảnh hưởng CER. Hai phiên Kaggle có thể khác driver/tải GPU. NVML có thể bỏ lỡ peak ngắn và gồm tiến trình khác. Pipeline Paddle có thể tải artifact qua registry khác với HF commit tham chiếu.",
               "", "## 9. Kết luận", "",
               f"Trên tập đã đo: CER tốt hơn: {table[0]['Better']}; WER tốt hơn: {table[1]['Better']}; p50 nhanh hơn: {table[3]['Better']}; peak VRAM thấp hơn: {table[6]['Better']}. Đây là kết luận theo từng tiêu chí, không tự suy ra một lựa chọn triển khai duy nhất. Khi triển khai thực tế, cần thêm dữ liệu đích, độ ổn định và chi phí vận hành."]
    (out / "comparison_report.md").write_text("\n".join(report), encoding="utf-8")
    # Deterministic top errors: largest best-model CER, then ID.
    examples = sorted((r for r in combined if max(r["paddle_cer"], r["deepseek_cer"]) > 0),
                      key=lambda r: (-min(r["paddle_cer"], r["deepseek_cer"]), r["sample_id"]))[:10]
    html_rows = ["<html><meta charset='utf-8'><style>body{font:16px sans-serif;max-width:1100px;margin:auto;background:#fff;color:#222}section{border-bottom:1px solid #ddd;padding:18px}img{max-width:420px}del{background:#fdd}ins{background:#dfd}pre{white-space:pre-wrap}</style><h1>Phân tích lỗi OCR</h1>"]
    manifest_path = paddle_dir / "manifest.csv"
    paths = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8", newline="") as fh:
            paths = {r["sample_id"]: Path(r["image_path"]) for r in csv.DictReader(fh)}
    for r in examples:
        pic = paths.get(r["sample_id"])
        if pic and not pic.is_file():
            pic = paddle_dir / "images" / pic.name
        img = ""
        if pic and pic.is_file():
            import base64
            mime = "image/jpeg" if pic.suffix.lower() in {".jpg",".jpeg"} else "image/png"
            img = f"<img src='data:{mime};base64,{base64.b64encode(pic.read_bytes()).decode()}'>"
        diff = lambda pred: "".join(("<del>" if x.startswith("-") else "<ins>" if x.startswith("+") else "<span>") + html.escape(x[2:]) + ("</del>" if x.startswith("-") else "</ins>" if x.startswith("+") else "</span>") for x in difflib.ndiff(r["ground_truth"], pred))
        flags = category_flags(r["ground_truth"], r["paddle_prediction"])
        note = ", ".join(k for k,v in flags.items() if v) or "Lỗi ký tự khác"
        html_rows.append(f"<section><h2>{html.escape(r['sample_id'])}</h2>{img}<p>GT: {html.escape(r['ground_truth'])}</p><p>Paddle CER {r['paddle_cer']:.3f}: {html.escape(r['paddle_prediction'])}</p><p>DeepSeek CER {r['deepseek_cer']:.3f}: {html.escape(r['deepseek_prediction'])}</p><p>Diff Paddle: {diff(r['paddle_prediction'])}</p><p>Diff DeepSeek: {diff(r['deepseek_prediction'])}</p><p>Nhận xét tự động: {html.escape(note)}</p></section>")
    html_rows.append("</html>")
    (out / "error_analysis.html").write_text("\n".join(html_rows), encoding="utf-8")
    return summary
