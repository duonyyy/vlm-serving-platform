"""Unit tests for InferenceCache and LRU Eviction."""

import time
import unittest
from packages.caching.cache import InferenceCache


class TestInferenceCache(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = InferenceCache(max_entries=3, default_ttl_sec=0.2)

    def test_cache_set_and_get(self) -> None:
        key = InferenceCache.generate_key(
            endpoint="/v1/chat/completions",
            model="qwen-2b",
            prompt="describe",
            image_bytes_or_url="data:image/jpeg;base64,abc",
        )
        self.assertIsNone(self.cache.get(key))
        self.assertEqual(self.cache.stats.misses, 1)

        val = {"choices": [{"message": {"content": "ok"}}], "timing": {"total_ms": 100.0}}
        self.cache.set(key, val)

        res = self.cache.get(key)
        self.assertIsNotNone(res)
        self.assertEqual(res["choices"][0]["message"]["content"], "ok")
        self.assertEqual(self.cache.stats.hits, 1)
        self.assertGreater(self.cache.stats.saved_latency_ms, 0)
        self.assertEqual(self.cache.stats.hit_rate, 0.5)

    def test_lru_eviction(self) -> None:
        """When cache exceeds max_entries, oldest entry is evicted."""
        for i in range(3):
            self.cache.set(f"key_{i}", {"idx": i})
        self.assertEqual(len(self.cache), 3)

        # Access key_0 to make it recently used
        self.cache.get("key_0")

        # Insert 4th element -> key_1 should be evicted (as key_0 was recently accessed)
        self.cache.set("key_3", {"idx": 3})
        self.assertEqual(len(self.cache), 3)
        self.assertIsNotNone(self.cache.get("key_0"))
        self.assertIsNone(self.cache.get("key_1"))
        self.assertEqual(self.cache.stats.evictions, 1)

    def test_ttl_expiration(self) -> None:
        """Entries expire after TTL seconds."""
        self.cache.set("expire_key", {"data": 123}, ttl_seconds=0.1)
        self.assertIsNotNone(self.cache.get("expire_key"))

        time.sleep(0.15)
        self.assertIsNone(self.cache.get("expire_key"))

    def test_deterministic_key_generation(self) -> None:
        k1 = InferenceCache.generate_key("/v1/extract", "qwen", "prompt", "url_1", {"t": 0})
        k2 = InferenceCache.generate_key("/v1/extract", "qwen", "prompt", "url_1", {"t": 0})
        k3 = InferenceCache.generate_key("/v1/extract", "qwen", "different", "url_1", {"t": 0})
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)


if __name__ == "__main__":
    unittest.main()
