"""Unit tests for DynamicResizer and ResolutionPolicy."""

import unittest
from PIL import Image

from packages.image_processing.dynamic_resizer import DynamicResizer, ResolutionPolicy


class TestDynamicResizer(unittest.TestCase):
    def test_visual_tokens_estimation(self) -> None:
        tokens = DynamicResizer.estimate_visual_tokens(1120, 1120, patch_size=28)
        self.assertEqual(tokens, 40 * 40)  # 1600 tokens

    def test_vram_estimation(self) -> None:
        vram = DynamicResizer.estimate_vram_mb(tokens=1000, base_vram_mb=4800.0)
        self.assertEqual(vram, 4800.0 + 850.0)

    def test_scaling_policy_768(self) -> None:
        resizer = DynamicResizer(ResolutionPolicy.LOW_768)
        img = Image.new("RGB", (1536, 1024), color="white")
        opt_img, meta = resizer.optimize_image(img)
        self.assertEqual(opt_img.size, (768, 512))
        self.assertTrue(meta["resized"])
        self.assertGreater(meta["token_reduction_pct"], 0)

    def test_scaling_policy_unbounded(self) -> None:
        resizer = DynamicResizer(ResolutionPolicy.UNBOUNDED)
        img = Image.new("RGB", (2000, 1500), color="white")
        opt_img, meta = resizer.optimize_image(img)
        self.assertEqual(opt_img.size, (2000, 1500))
        self.assertFalse(meta["resized"])
        self.assertEqual(meta["token_reduction_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
