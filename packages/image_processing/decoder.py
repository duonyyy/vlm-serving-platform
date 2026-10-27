"""Unified image decoding from URLs, Base64 data URIs, raw bytes, and files."""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any
import httpx
from PIL import Image

from .safety import is_safe_url
from .validator import MAX_IMAGE_BYTES, validate_image_bytes

DATA_URI_PATTERN = re.compile(r"^data:image\/(jpeg|jpg|png|webp);base64,(.+)$", re.IGNORECASE | re.DOTALL)


def decode_image(
    image_input: str | bytes | Path,
    allow_remote_urls: bool = True,
    url_allowlist: list[str] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Decode and validate image input into a Pillow Image and metadata.

    Args:
        image_input: Base64 data URI, raw base64 string, URL, raw bytes, or Path.
        allow_remote_urls: Whether to permit downloading from HTTP/HTTPS.
        url_allowlist: Optional domain allowlist for SSRF restriction.

    Returns:
        (PIL.Image, metadata)

    Raises:
        ValueError: If input format is invalid, unsafe, or corrupted.
    """
    raw_bytes: bytes

    if isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        fmt = image_input.format or "PNG"
        if fmt.upper() not in {"JPEG", "JPG", "PNG", "WEBP"}:
            fmt = "PNG"
        image_input.save(buf, format=fmt)
        raw_bytes = buf.getvalue()

    elif isinstance(image_input, bytes):
        raw_bytes = image_input

    elif isinstance(image_input, Path) or (isinstance(image_input, str) and (Path(image_input).is_file() if len(image_input) < 1024 else False)):
        file_path = Path(image_input)
        if not file_path.is_file():
            raise ValueError(f"Image file does not exist: {file_path}")
        raw_bytes = file_path.read_bytes()

    elif isinstance(image_input, str):
        image_input_clean = image_input.strip()

        # Case 1: Data URI (e.g. data:image/png;base64,iVBOR...)
        match = DATA_URI_PATTERN.match(image_input_clean)
        if match:
            b64_data = match.group(2).strip()
            try:
                raw_bytes = base64.b64decode(b64_data, validate=True)
            except Exception as exc:
                raise ValueError(f"Invalid base64 payload in data URI: {exc}") from exc

        # Case 2: HTTP/HTTPS remote URL
        elif image_input_clean.startswith(("http://", "https://")):
            if not allow_remote_urls:
                raise ValueError("Remote URL downloading is disabled on this worker")

            safe, reason = is_safe_url(image_input_clean, allowlist=url_allowlist)
            if not safe:
                raise ValueError(f"URL security check failed: {reason}")

            try:
                with httpx.Client(timeout=10.0, follow_redirects=False) as client:
                    resp = client.get(image_input_clean)
                    resp.raise_for_status()
                    raw_bytes = resp.content
            except Exception as exc:
                raise ValueError(f"Failed to fetch remote image from URL: {exc}") from exc

        # Case 3: Raw base64 string without data: prefix
        else:
            try:
                raw_bytes = base64.b64decode(image_input_clean, validate=True)
            except Exception as exc:
                raise ValueError("String is neither a valid data URI, file path, URL, nor raw base64") from exc

    else:
        raise ValueError(f"Unsupported image input type: {type(image_input).__name__}")

    # Validate byte payload against safety thresholds
    meta = validate_image_bytes(raw_bytes)

    # Open verified image in memory
    image = Image.open(io.BytesIO(raw_bytes))
    return image, meta
