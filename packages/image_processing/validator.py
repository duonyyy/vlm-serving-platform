"""Validation of raw image bytes, dimensions, and formats."""

from __future__ import annotations

import hashlib
import io
from typing import Any
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MiB
MAX_PIXELS = 20_000_000  # 20 Megapixels
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# Set PIL decompression bomb threshold
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def validate_image_bytes(raw_bytes: bytes) -> dict[str, Any]:
    """Validate raw image bytes against byte limit, pixel limit, and format rules.

    Returns:
        Dict with image format, byte size, width, height, sha256 hash.
    Raises:
        ValueError: If the image violates limits, is corrupted, or has unsupported format.
    """
    if not raw_bytes:
        raise ValueError("Image byte payload is empty")

    byte_size = len(raw_bytes)
    if byte_size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image size {byte_size} bytes exceeds limit of {MAX_IMAGE_BYTES} bytes (20 MiB)")

    digest = hashlib.sha256(raw_bytes).hexdigest()

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img_format = img.format
            if img_format not in ALLOWED_FORMATS:
                raise ValueError(f"Unsupported image format: '{img_format}'. Supported formats: {sorted(ALLOWED_FORMATS)}")

            width, height = img.size
            total_pixels = width * height
            if total_pixels > MAX_PIXELS:
                raise ValueError(
                    f"Image dimensions ({width}x{height} = {total_pixels} pixels) exceed safety limit of {MAX_PIXELS} pixels"
                )

            # Deep verify checks file integrity for corruption
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Failed to identify or decode valid image bytes: {exc}") from exc

    return {
        "format": img_format,
        "byte_size": byte_size,
        "width": width,
        "height": height,
        "sha256": digest,
    }
