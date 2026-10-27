"""Shared, fail-closed dataset and measurement code for Kaggle OCR notebooks."""
from __future__ import annotations

import base64
import csv
import gc
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import unicodedata
import urllib.request
from pathlib import Path
from statistics import median
from typing import Any, Callable

DATASET_ID = "wulipc/CC-OCR"
DATASET_FILE = "multi_lan_ocr/Vietnamese/Vietnamese_150.tsv"
EXPECTED_COLUMNS = {"index", "image", "question", "answer"}
MODEL_IDS = {"paddle": "PaddlePaddle/PaddleOCR-VL-1.6", "deepseek": "deepseek-ai/DeepSeek-OCR-2"}
TERMINAL = {"completed", "failed", "oom", "timeout"}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment() -> dict[str, Any]:
    import psutil
    import torch
    result: dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "disk_free_bytes": shutil.disk_usage("/kaggle/working").free,
        "gpu": [], "internet": False,
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            result["gpu"].append({"index": i, "name": p.name, "compute_capability": list(p.major_minor) if hasattr(p, "major_minor") else [p.major, p.minor], "total_vram_bytes": total, "free_vram_bytes": free})
    try:
        result["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["nvidia_smi_error"] = str(exc)
    try:
        with urllib.request.urlopen("https://huggingface.co", timeout=10) as response:
            result["internet"] = response.status == 200
    except Exception as exc:
        result["internet_error"] = str(exc)
    return result


def require_t4(env: dict[str, Any]) -> None:
    if not env["cuda_available"] or not env["gpu"]:
        raise RuntimeError("GPU CUDA không khả dụng; dừng benchmark GPU.")
    gpu = env["gpu"][0]
    if "T4" not in gpu["name"] or gpu["compute_capability"] != [7, 5]:
        raise RuntimeError(f"Cần NVIDIA T4 trên cuda:0, thực tế: {gpu}")


def download_tsv(workdir: Path, revision: str | None = None) -> tuple[Path, str]:
    from huggingface_hub import HfApi, hf_hub_download
    revision = revision or HfApi().dataset_info(DATASET_ID).sha
    path = Path(hf_hub_download(repo_id=DATASET_ID, repo_type="dataset", filename=DATASET_FILE, revision=revision, local_dir=workdir / "hf_dataset"))
    return path, revision


def inspect_tsv(path: Path) -> dict[str, Any]:
    """Read actual schema before decoding. Never print Base64 image data."""
    csv.field_size_limit(2**31 - 1)  # Base64 image field can be megabytes.
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = reader.fieldnames or []
        if not EXPECTED_COLUMNS.issubset(fields):
            raise ValueError(f"TSV schema mismatch: {fields}; required {sorted(EXPECTED_COLUMNS)}")
        preview = []
        for row in reader:
            preview.append({"index": row.get("index"), "question": row.get("question"), "answer": row.get("answer"), "image_chars": len(row.get("image") or "")})
            if len(preview) == 5:
                break
    return {"columns": fields, "preview": preview, "tsv_sha256": sha256(path)}


def make_manifest(path: Path, outdir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    from PIL import Image, UnidentifiedImageError
    csv.field_size_limit(2**31 - 1)
    images = outdir / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not EXPECTED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("TSV schema changed; inspect before parsing")
        for ordinal, item in enumerate(reader):
            sample_id = str(item.get("index", "")).strip()
            reason = ""
            try:
                if not sample_id or sample_id in seen_ids:
                    raise ValueError("empty or duplicate sample ID")
                seen_ids.add(sample_id)
                answer = item.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("empty ground truth")
                encoded = item.get("image")
                if not isinstance(encoded, str) or not encoded.strip():
                    raise ValueError("empty image")
                if encoded.startswith("data:"):
                    encoded = encoded.split(",", 1)[1]
                payload = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(payload)) as img:
                    img.verify()
                with Image.open(io.BytesIO(payload)) as img:
                    width, height = img.size
                    fmt = (img.format or "PNG").lower()
                    if width < 1 or height < 1:
                        raise ValueError("invalid dimensions")
                digest = hashlib.sha256(payload).hexdigest()
                suffix = ".jpg" if fmt in {"jpeg", "jpg"} else f".{fmt}"
                image_path = images / f"{ordinal:03d}{suffix}"
                image_path.write_bytes(payload)
                rows.append({"sample_id": sample_id, "ordinal": str(ordinal), "image_path": str(image_path), "image_sha256": digest,
                             "width": str(width), "height": str(height), "ground_truth": answer, "question": item.get("question", ""),
                             "duplicate_of": seen_hashes.get(digest, "")})
                seen_hashes.setdefault(digest, sample_id)
            except (ValueError, TypeError, KeyError, UnidentifiedImageError, OSError, base64.binascii.Error) as exc:
                reason = str(exc)
            if reason:
                rejected.append({"ordinal": str(ordinal), "sample_id": sample_id, "reason": reason})
    if not rows:
        raise RuntimeError("No valid samples")
    fields = list(rows[0])
    with (outdir / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with (outdir / "rejected_samples.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ordinal", "sample_id", "reason"])
        writer.writeheader(); writer.writerows(rejected)
    return rows, rejected


def manifest_fingerprint(rows: list[dict[str, str]]) -> str:
    canonical = [{k: row[k] for k in ("sample_id", "image_sha256", "width", "height", "ground_truth")}
                 for row in rows]
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize(text: str, mode: str) -> str:
    if mode == "strict":
        return text
    if mode != "normalized":
        raise ValueError(mode)
    return " ".join(unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n").split())


def plain_text(raw: str) -> str:
    """Conservative Markdown removal; keep visible characters and punctuation."""
    raw = re.sub(r"!\[[^]]*\]\([^)]*\)", "", raw)
    raw = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", raw)
    raw = re.sub(r"(?m)^\s{0,3}(?:#{1,6}\s+|>\s+|[-*+]\s+|\d+\.\s+)", "", raw)
    raw = raw.replace("**", "").replace("__", "").replace("`", "")
    return raw


def _distance(a: list[str] | str, b: list[str] | str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        previous = list(range(len(b) + 1))
        for i, left in enumerate(a, start=1):
            current = [i]
            for j, right in enumerate(b, start=1):
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
            previous = current
        return previous[-1]


def score(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if not records:
        return {"status": "NOT_EVALUATED", "n": 0}
    chars_ref = chars_edit = words_ref = words_edit = exact = empty = failed = 0
    per_image = []
    for row in records:
        gt = normalize(row["ground_truth"], mode)
        ok = row["status"] == "completed"
        pred = normalize(row.get("prediction_text", "") if ok else "", mode)
        failed += not ok
        empty += not pred
        exact += pred == gt and ok
        cd = _distance(gt, pred)
        wd = _distance(gt.split(), pred.split())
        chars_ref += len(gt); chars_edit += cd
        words_ref += len(gt.split()); words_edit += wd
        per_image.append(cd / max(1, len(gt)))
    n = len(records)
    return {"status": "MEASURED", "n": n, "corpus_cer": chars_edit / max(1, chars_ref),
            "corpus_whitespace_wer": words_edit / max(1, words_ref), "median_per_image_cer": median(per_image),
            "exact_match_rate": exact / n, "empty_prediction_rate": empty / n, "runtime_error_rate": failed / n,
            "failed_policy": "failed/oom/timeout => empty prediction for end-to-end quality; reported separately"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        fh.flush(); os.fsync(fh.fileno())


class DeviceVRAMSampler:
    """Sample whole-device NVML usage; includes other processes and may miss short spikes."""
    def __init__(self) -> None:
        self.peak: int | None = None
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.pynvml = None

    def start(self) -> None:
        if self.pynvml is None:
            return
        def poll() -> None:
            while not self.stop.is_set():
                try:
                    used = int(self.pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)
                    self.peak = max(self.peak or 0, used)
                except Exception:
                    pass
                self.stop.wait(.05)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()

    def finish(self) -> int | None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
        return self.peak


def export_predictions(rows: list[dict[str, Any]], outdir: Path) -> None:
    fields = ["sample_id", "image_sha256", "ground_truth", "status", "prediction_raw", "prediction_text", "latency_s", "peak_allocated_bytes", "peak_reserved_bytes", "peak_device_vram_bytes", "ram_rss_bytes", "truncated", "error"]
    with (outdir / "predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    with (outdir / "failures.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(r for r in rows if r["status"] != "completed")


def performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    good = [float(r["latency_s"]) for r in rows if r["status"] == "completed"]
    all_times = [float(r["latency_s"]) for r in rows]
    peaks = [int(r["peak_device_vram_bytes"]) for r in rows if r.get("peak_device_vram_bytes") is not None]
    return {"completed_count": len(good), "total_inference_seconds": sum(all_times),
            "throughput_images_per_second": len(good) / sum(all_times) if all_times else None,
            "latency_p50_s": float(np.percentile(good, 50)) if good else None,
            "latency_p90_s": float(np.percentile(good, 90)) if good else None,
            "latency_p95_s": float(np.percentile(good, 95)) if good else None,
            "peak_device_vram_bytes": max(peaks) if peaks else None,
            "vram_note": "NVML whole-device sampled at 50 ms; includes other processes and can miss brief peaks"}


def run_benchmark(manifest: list[dict[str, str]], outdir: Path, infer: Callable[[Path], tuple[str, bool | None]],
                  sync: Callable[[], None], gpu_peak: Callable[[], tuple[int | None, int | None, int | None]],
                  mode: str, warmup: Path, max_seconds: float = 300.0,
                  reset_peak: Callable[[], None] | None = None) -> list[dict[str, Any]]:
    """Append a terminal record after each sample. Warm-up is excluded from measured rows."""
    import psutil
    outdir.mkdir(parents=True, exist_ok=True)
    chosen = manifest[:5] if mode == "SMOKE_TEST" else manifest
    path = outdir / "predictions.jsonl"
    old = read_jsonl(path)
    ids = [(x["sample_id"], x["image_sha256"]) for x in chosen]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate sample keys")
    allowed = set(ids)
    if any((r["sample_id"], r["image_sha256"]) not in allowed for r in old):
        raise RuntimeError("checkpoint contains different dataset/mode")
    if len({(r["sample_id"], r["image_sha256"]) for r in old}) != len(old):
        raise RuntimeError("duplicate checkpoint rows; resolve manually")
    if any(r.get("status") not in TERMINAL for r in old):
        raise RuntimeError("checkpoint contains unknown status")
    done = {(r["sample_id"], r["image_sha256"]) for r in old if r["status"] in TERMINAL}
    if len(done) < len(chosen):
        sync(); start = time.perf_counter()
        infer(warmup); sync()
        warmup_result = {"image_path": str(warmup), "seconds": time.perf_counter() - start,
                         "resume": bool(old)}
        write_json(outdir / "warmup.json", warmup_result)
        append_jsonl(outdir / "warmup_history.jsonl", warmup_result)
    for item in chosen:
        key = (item["sample_id"], item["image_sha256"])
        if key in done:
            continue
        sync()
        if reset_peak:
            reset_peak()
        sampler = DeviceVRAMSampler()
        sampler.start()
        start = time.perf_counter()
        record: dict[str, Any] = {"sample_id": key[0], "image_sha256": key[1], "ground_truth": item["ground_truth"], "status": "failed", "prediction_raw": "", "prediction_text": "", "truncated": False, "error": ""}
        try:
            raw, truncated = infer(Path(item["image_path"]))
            if not isinstance(raw, str):
                raise TypeError(f"inference returned {type(raw).__name__}; expected str")
            sync()
            record.update(status="completed", prediction_raw=raw, prediction_text=plain_text(raw), truncated=truncated)
        except Exception as exc:
            sync()
            record.update(status="oom" if "out of memory" in str(exc).lower() else "failed", error=f"{type(exc).__name__}: {exc}"[:2000])
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
        elapsed = time.perf_counter() - start
        device_peak = sampler.finish()
        if elapsed > max_seconds and record["status"] == "completed":
            record["status"] = "timeout"  # post hoc threshold; no hard preemption
        a, r, d = gpu_peak()
        record.update(latency_s=elapsed, peak_allocated_bytes=a, peak_reserved_bytes=r, peak_device_vram_bytes=device_peak or d,
                      ram_rss_bytes=psutil.Process().memory_info().rss)
        append_jsonl(path, record)
        done.add(key)
        if record["status"] == "oom":
            break  # repeated OOM is unsafe; preserve checkpoint
    rows = read_jsonl(path)
    export_predictions(rows, outdir)
    write_json(outdir / "metrics.json", {"run_status": "MEASURED" if len(rows) == len(chosen) else "PARTIAL",
                                         "expected_samples": len(chosen), "recorded_samples": len(rows),
                                         **{m: score(rows, m) for m in ("strict", "normalized")},
                                         "performance": performance(rows)})
    return rows
