"""Single-GPU Qwen3-VL Transformers baseline for Kaggle T4."""

from __future__ import annotations

import time
import re
from pathlib import Path

from .core import MODEL_ID

MAX_IMAGE_EDGE = 1280


def prepare_image(image_path: Path, max_edge: int = MAX_IMAGE_EDGE):
    """Apply the recorded T4 image-size policy without changing source bytes."""
    from PIL import Image, ImageOps

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    oriented_size = image.size
    if max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return image, oriented_size


class QwenBaseline:
    resize_longest_edge = MAX_IMAGE_EDGE

    def __init__(self, revision: str | None = None) -> None:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7, 5):
            raise RuntimeError("Qwen baseline requires NVIDIA T4 at cuda:0")
        self.torch = torch
        self.revision = revision or HfApi().model_info(MODEL_ID).sha
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", self.revision):
            raise ValueError("model revision must resolve to a 40-character commit SHA")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=self.revision)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=self.revision, dtype=torch.float16,
            attn_implementation="eager", device_map="cuda:0"
        ).eval()
        # CPU offload makes T4 latency and VRAM comparisons misleading.
        placement = getattr(self.model, "hf_device_map", None)
        if placement and any(str(device) not in {"0", "cuda:0", "cuda"} for device in placement.values()):
            raise RuntimeError(f"model was offloaded outside cuda:0: {placement}")
        if str(self.model.device) != "cuda:0":
            raise RuntimeError(f"model did not load on cuda:0: {self.model.device}")

    def infer(self, image_path: Path, prompt: str, max_new_tokens: int = 128) -> dict:
        torch = self.torch
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        image, oriented_size = prepare_image(image_path, self.resize_longest_edge)
        processed_size = image.size
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        torch.cuda.synchronize(0)
        preprocessing_ms = (time.perf_counter() - start) * 1000
        start_infer = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize(0)
        inference_ms = (time.perf_counter() - start_infer) * 1000
        input_tokens = inputs["input_ids"].shape[-1]
        output_ids = generated[0][input_tokens:]
        text = self.processor.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return {"prediction_text": text, "input_tokens": int(input_tokens),
                "output_tokens": int(output_ids.shape[-1]),
                "oriented_width": oriented_size[0], "oriented_height": oriented_size[1],
                "processed_width": processed_size[0], "processed_height": processed_size[1],
                "preprocessing_ms": round(preprocessing_ms, 3),
                "inference_ms": round(inference_ms, 3),
                "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved(0))}
