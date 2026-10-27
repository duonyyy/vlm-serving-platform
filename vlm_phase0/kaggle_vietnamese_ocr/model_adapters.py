"""Official-checkpoint adapters. DeepSeek T4 patch is explicit and revision-bound."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PADDLE_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
DEEPSEEK_ID = "deepseek-ai/DeepSeek-OCR-2"
DEEPSEEK_PROMPT = "<image>\nFree OCR. "
MAX_NEW_TOKENS = 2048


def install_common() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                           "huggingface_hub==0.36.0", "Pillow==11.3.0", "psutil==7.0.0", "rapidfuzz==3.14.1", "pandas==2.2.3", "matplotlib==3.10.3", "nvidia-ml-py==12.575.51"])


def install_paddle(env: dict[str, Any]) -> None:
    """Match an official CUDA wheel to current runtime; stop for unknown combinations."""
    cuda = env.get("torch_cuda_runtime") or ""
    major = cuda.split(".")[0]
    if major == "12":
        index = "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
        if int(cuda.split(".")[1]) < 6:
            raise RuntimeError("CUDA <12.6: choose an official Paddle wheel manually; notebook will not guess")
    else:
        raise RuntimeError(f"No audited Paddle wheel for Kaggle torch CUDA {cuda}; inspect official install matrix")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paddlepaddle-gpu==3.2.1", "-i", index])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'paddleocr[doc-parser]==3.7.0'])


class PaddleAdapter:
    def __init__(self) -> None:
        from paddleocr import PaddleOCRVL
        self.init_start = time.perf_counter()
        self.pipeline = PaddleOCRVL(pipeline_version="v1.6", device="gpu:0", use_layout_detection=True,
                                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                                    use_queues=False, markdown_ignore_labels=[])
        self.init_seconds = time.perf_counter() - self.init_start

    def infer(self, image: Path) -> tuple[str, None]:
        results = list(self.pipeline.predict(str(image), max_new_tokens=MAX_NEW_TOKENS))
        if len(results) != 1:
            raise RuntimeError(f"Expected one page result; got {len(results)}")
        md = results[0].markdown
        if not isinstance(md, dict) or "markdown_texts" not in md:
            raise TypeError(f"Unexpected Paddle markdown structure: {type(md).__name__}")
        texts = md["markdown_texts"]
        if isinstance(texts, str):
            return texts, None  # Pipeline does not expose a reliable truncation flag.
        if isinstance(texts, list) and all(isinstance(x, str) for x in texts):
            return "\n".join(texts), None
        raise TypeError(f"Unexpected markdown_texts: {type(texts).__name__}")

    @staticmethod
    def sync() -> None:
        import paddle
        paddle.device.cuda.synchronize()

    @staticmethod
    def gpu_peak() -> tuple[int | None, int | None, None]:
        import paddle
        api = paddle.device.cuda
        try:
            return int(api.max_memory_allocated(0)), int(api.max_memory_reserved(0)), None
        except (AttributeError, RuntimeError, TypeError):
            return None, None, None

    @staticmethod
    def reset_peak() -> None:
        import paddle
        try:
            paddle.device.cuda.reset_peak_memory_stats(0)
        except (AttributeError, RuntimeError, TypeError):
            pass


def patch_deepseek_for_t4(snapshot: Path, max_new_tokens: int = MAX_NEW_TOKENS) -> dict[str, Any]:
    """Patch only known hard-coded inference choices, with exact-count guards."""
    target = snapshot / "modeling_deepseekocr2.py"
    source = target.read_text(encoding="utf-8")
    patch_log = snapshot / "t4_patch.json"
    if patch_log.exists():
        previous = json.loads(patch_log.read_text(encoding="utf-8"))
        if hashlib.sha256(source.encode()).hexdigest() == previous.get("patched_sha256"):
            return previous
    original_hash = hashlib.sha256(source.encode()).hexdigest()
    # Five executable BF16 sites plus one comment in the reviewed upstream file.
    if source.count("torch.bfloat16") != 6 or source.count("max_new_tokens=8192") != 2:
        raise RuntimeError("DeepSeek official source changed; audit T4 patch before continuing")
    source = source.replace("torch.bfloat16", "torch.float16")
    source = source.replace("max_new_tokens=8192", f"max_new_tokens={max_new_tokens}")
    target.write_text(source, encoding="utf-8")
    result = {"file": target.name, "original_sha256": original_hash,
            "patched_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "changes": ["BF16 tensors and autocast -> FP16", f"max_new_tokens 8192 -> {max_new_tokens}"],
            "risk": "FP16 is an unvalidated precision port; output quality may differ from official BF16 baseline"}
    patch_log.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


class DeepSeekAdapter:
    def __init__(self, cache_dir: Path, revision: str | None = None) -> None:
        import torch
        from huggingface_hub import HfApi, snapshot_download
        from transformers import AutoModel, AutoTokenizer
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7, 5):
            raise RuntimeError("DeepSeek T4 path requires CC 7.5")
        self.revision = revision or HfApi().model_info(DEEPSEEK_ID).sha
        self.snapshot = Path(snapshot_download(repo_id=DEEPSEEK_ID, revision=self.revision, local_dir=cache_dir / "deepseek_model"))
        self.patch = patch_deepseek_for_t4(self.snapshot)
        self.init_start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), trust_remote_code=True, local_files_only=True)
        # torch_dtype applies while loading; avoid a transient FP32 GPU copy.
        self.model = AutoModel.from_pretrained(str(self.snapshot), trust_remote_code=True, local_files_only=True,
                                               use_safetensors=True, torch_dtype=torch.float16,
                                               _attn_implementation="eager", low_cpu_mem_usage=True).eval().to("cuda:0")
        if self.model.config._attn_implementation != "eager":
            raise RuntimeError("eager attention was not selected")
        self.init_seconds = time.perf_counter() - self.init_start

    def infer(self, image: Path) -> tuple[str, bool]:
        # eval_mode returns decoded text according to the official source; save_results=False avoids eval() branches.
        text = self.model.infer(self.tokenizer, prompt=DEEPSEEK_PROMPT, image_file=str(image),
                                output_path="/kaggle/working/deepseek_tmp", base_size=1024,
                                image_size=768, crop_mode=True, save_results=False, eval_mode=True)
        if not isinstance(text, str):
            raise TypeError(f"model.infer returned {type(text).__name__}")
        # Exact termination is unavailable through infer(); report a conservative flag.
        truncated = len(self.tokenizer.encode(text, add_special_tokens=False)) >= MAX_NEW_TOKENS
        return text, truncated

    @staticmethod
    def sync() -> None:
        import torch
        torch.cuda.synchronize(0)

    @staticmethod
    def gpu_peak() -> tuple[int, int, None]:
        import torch
        return torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0), None

    @staticmethod
    def reset_peak() -> None:
        import torch
        torch.cuda.reset_peak_memory_stats(0)
