"""Phase 0 data validation and evidence capture; no model imports at module load."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MIN_CASES = 20
CATEGORIES = {"scene", "document", "text", "chart"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 20_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: Path, dataset_root: Path) -> list[dict]:
    """Require 20 distinct local images with review notes across four strata."""
    from PIL import Image, UnidentifiedImageError

    root = dataset_root.absolute()
    if not root.is_dir():
        raise ValueError("dataset root does not exist")
    cases: list[dict] = []
    ids: set[str] = set()
    hashes: set[str] = set()
    with manifest.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                case_id = row["sample_id"]
                relative = Path(row["image_path"])
                prompt = row["prompt"]
                category = row["category"]
                expected = row["expected_observation"]
                if not all(isinstance(x, str) and x.strip() for x in (case_id, prompt, expected)):
                    raise ValueError("sample_id, prompt and expected_observation must be nonempty strings")
                if case_id in ids:
                    raise ValueError("duplicate sample_id")
                if category not in CATEGORIES:
                    raise ValueError(f"category must be one of {sorted(CATEGORIES)}")
                if relative.is_absolute() or relative.drive or ".." in relative.parts:
                    raise ValueError("image_path must be relative to dataset root")
                image_path = root / relative
                # Reject symlink hops without relying on realpath, which is unavailable
                # for some sandbox-mounted Windows directories.
                if any((root.joinpath(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
                    raise ValueError("symlinked image path is not allowed")
                if not image_path.is_file():
                    raise ValueError("image_path escapes dataset root or is not a file")
                if image_path.stat().st_size > MAX_IMAGE_BYTES:
                    raise ValueError("image exceeds byte limit")
                with Image.open(image_path) as image:
                    if image.width * image.height > MAX_PIXELS:
                        raise ValueError("image exceeds pixel limit")
                    if image.format not in {"JPEG", "PNG", "WEBP"}:
                        raise ValueError("unsupported image format")
                    width, height = image.size
                    image.verify()
                digest = sha256(image_path)
                if digest in hashes:
                    raise ValueError("duplicate image bytes")
                ids.add(case_id)
                hashes.add(digest)
                cases.append({"sample_id": case_id, "image_path": str(image_path),
                              "image_sha256": digest, "image_bytes": image_path.stat().st_size,
                              "width": width, "height": height, "category": category,
                              "prompt": prompt, "expected_observation": expected})
            except (KeyError, TypeError, ValueError, OSError, UnidentifiedImageError, json.JSONDecodeError) as exc:
                raise ValueError(f"manifest line {lineno}: {exc}") from exc
    if len(cases) < MIN_CASES:
        raise ValueError(f"at least {MIN_CASES} valid, distinct images required; found {len(cases)}")
    counts = Counter(case["category"] for case in cases)
    underrepresented = {category: counts[category] for category in CATEGORIES if counts[category] < 5}
    if underrepresented:
        raise ValueError(f"at least 5 images per category required: {underrepresented}")
    return cases


def hardware_report() -> dict:
    """Collect measurements from this runtime; unavailable fields remain explicit."""
    report = {"status": "MEASURED", "captured_at_utc": utc_now(),
              "python": sys.version, "platform": platform.platform(),
              "packages": {}, "gpu": [], "kaggle_quota": "USER_RECORDED_SEPARATELY"}
    for package in ("torch", "transformers", "accelerate", "Pillow", "huggingface-hub"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        report["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else None
        report["nvidia_smi_error"] = result.stderr.strip() if result.returncode else None
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi"] = None
        report["nvidia_smi_error"] = str(exc)
    try:
        import torch
        report["torch_cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                report["gpu"].append({"index": index, "name": properties.name,
                                      "compute_capability": list(torch.cuda.get_device_capability(index)),
                                      "total_vram_bytes": properties.total_memory})
    except ImportError:
        report["torch_cuda_runtime"] = None
        report["cuda_available"] = False
    return report


def require_t4(report: dict) -> None:
    if not report["gpu"] or "T4" not in report["gpu"][0]["name"] or report["gpu"][0]["compute_capability"] != [7, 5]:
        raise RuntimeError("Phase 0 Kaggle baseline requires an NVIDIA T4 at cuda:0; see hardware_report.json")
