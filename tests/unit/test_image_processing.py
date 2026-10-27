"""Unit tests for image decoding, validation, resizing, and safety checks."""

import base64
import io
import unittest
from PIL import Image
from packages.image_processing.decoder import decode_image
from packages.image_processing.resizer import prepare_image
from packages.image_processing.safety import is_safe_url
from packages.image_processing.validator import (
    ALLOWED_FORMATS,
    MAX_IMAGE_BYTES,
    MAX_PIXELS,
    validate_image_bytes,
)


def create_test_image_bytes(format: str = "PNG", size: tuple[int, int] = (100, 100), color: str = "blue") -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


class ImageProcessingTests(unittest.TestCase):
    def test_validate_valid_formats(self) -> None:
        for fmt in ["PNG", "JPEG", "WEBP"]:
            raw = create_test_image_bytes(fmt, size=(50, 50))
            meta = validate_image_bytes(raw)
            self.assertEqual(meta["format"], fmt)
            self.assertEqual(meta["width"], 50)
            self.assertEqual(meta["height"], 50)
            self.assertIn("sha256", meta)

    def test_unsupported_format_rejected(self) -> None:
        # BMP format should be rejected
        raw_bmp = create_test_image_bytes("BMP", size=(50, 50))
        with self.assertRaises(ValueError) as ctx:
            validate_image_bytes(raw_bmp)
        self.assertIn("Unsupported image format", str(ctx.exception))

    def test_corrupted_bytes_rejected(self) -> None:
        corrupt = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"random_garbage_bytes_here"
        with self.assertRaises(ValueError) as ctx:
            validate_image_bytes(corrupt)
        self.assertIn("Failed to identify or decode", str(ctx.exception))

    def test_empty_bytes_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_image_bytes(b"")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_byte_size_limit_rejected(self) -> None:
        fake_oversize = b"0" * (MAX_IMAGE_BYTES + 10)
        with self.assertRaises(ValueError) as ctx:
            validate_image_bytes(fake_oversize)
        self.assertIn("exceeds limit", str(ctx.exception).lower())

    def test_pixel_limit_rejected(self) -> None:
        # Create an image header claiming huge dimensions > 20 MP without generating 20M physical pixels
        # A 5000x5000 image has 25,000,000 pixels > MAX_PIXELS (20,000,000)
        raw_large = create_test_image_bytes("JPEG", size=(5000, 5000))
        with self.assertRaises(ValueError) as ctx:
            validate_image_bytes(raw_large)
        self.assertIn("exceed safety limit", str(ctx.exception).lower())

    def test_decode_image_from_data_uri(self) -> None:
        raw = create_test_image_bytes("JPEG", size=(80, 60))
        b64 = base64.b64encode(raw).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64}"

        img, meta = decode_image(data_uri)
        self.assertEqual(meta["format"], "JPEG")
        self.assertEqual(img.size, (80, 60))

    def test_resizing_longest_edge_policy(self) -> None:
        # Large image (2000 x 1000) should be resized to max 1280
        large_img = Image.new("RGB", (2000, 1000), color="green")
        processed, orig_size, proc_size = prepare_image(large_img, max_edge=1280)
        self.assertEqual(orig_size, (2000, 1000))
        self.assertEqual(proc_size, (1280, 640))
        self.assertEqual(max(proc_size), 1280)

        # Image within 1280 should NOT be resized
        small_img = Image.new("RGB", (800, 600), color="yellow")
        processed_small, orig_small, proc_small = prepare_image(small_img, max_edge=1280)
        self.assertEqual(proc_small, (800, 600))

    def test_ssrf_safety_blocks_private_networks(self) -> None:
        bad_urls = [
            "http://127.0.0.1:8000/test.png",
            "http://localhost:8080/secret",
            "http://10.0.0.5/invoice.jpg",
            "http://192.168.1.100/admin.png",
            "http://169.254.169.254/latest/meta-data/",
            "ftp://example.com/image.png",
        ]
        for url in bad_urls:
            safe, reason = is_safe_url(url)
            self.assertFalse(safe, f"Expected {url} to be blocked, but got safe: {reason}")


if __name__ == "__main__":
    unittest.main()
