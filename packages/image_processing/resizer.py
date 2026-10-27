"""Image normalization and resizing according to Phase 0 policy."""

from __future__ import annotations

from PIL import Image, ImageOps

DEFAULT_MAX_EDGE = 1280


def prepare_image(
    image: Image.Image,
    max_edge: int = DEFAULT_MAX_EDGE,
) -> tuple[Image.Image, tuple[int, int], tuple[int, int]]:
    """Orient and resize image according to the recorded T4 policy without changing source.

    1. Fix EXIF orientation and ensure RGB format.
    2. Resize if the longest edge exceeds `max_edge`, preserving exact aspect ratio using LANCZOS.

    Returns:
        (processed_image, oriented_size, processed_size)
    """
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    oriented_size = oriented.size

    if max(oriented.size) > max_edge:
        # Create a copy to resize without mutating the input object
        processed = oriented.copy()
        processed.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    else:
        processed = oriented

    processed_size = processed.size
    return processed, oriented_size, processed_size
