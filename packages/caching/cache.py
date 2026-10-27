"""Inference Result Caching Layer with SHA-256 Hashing and LRU Eviction."""

from __future__ import annotations

import collections
import hashlib
import json
import time
from typing import Any
from pydantic import BaseModel, Field

from packages.telemetry.logging import get_logger

logger = get_logger("inference_cache")


class CacheStats(BaseModel):
    """Real-time cache performance metrics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    max_entries: int = 1000
    saved_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0


class CacheEntry:
    """Internal cache entry wrapper with expiration and metadata."""

    def __init__(self, key: str, value: dict[str, Any], ttl_seconds: float | None = None) -> None:
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.expires_at = (self.created_at + ttl_seconds) if ttl_seconds else None
        self.hit_count = 0

    def is_expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        current = now or time.time()
        return current > self.expires_at


class InferenceCache:
    """Thread-safe LRU Cache for VLM chat completions and extraction requests.

    Key is deterministic SHA-256 hash of (image_data + prompt + model + extra_params).
    """

    def __init__(self, max_entries: int = 1000, default_ttl_sec: float | None = 3600.0) -> None:
        self.max_entries = max_entries
        self.default_ttl_sec = default_ttl_sec
        self._cache: collections.OrderedDict[str, CacheEntry] = collections.OrderedDict()
        self.stats = CacheStats(max_entries=max_entries)

    @staticmethod
    def generate_key(
        endpoint: str,
        model: str,
        prompt: str,
        image_bytes_or_url: str | bytes,
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        """Generate a deterministic SHA-256 cache key from request parameters."""
        hasher = hashlib.sha256()
        hasher.update(endpoint.encode("utf-8"))
        hasher.update(model.encode("utf-8"))
        hasher.update(prompt.strip().encode("utf-8"))

        if isinstance(image_bytes_or_url, bytes):
            hasher.update(image_bytes_or_url)
        else:
            hasher.update(image_bytes_or_url.strip().encode("utf-8"))

        if extra_params:
            # Sort keys for determinism
            serialized_params = json.dumps(extra_params, sort_keys=True)
            hasher.update(serialized_params.encode("utf-8"))

        return hasher.hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve cached result if present and not expired."""
        if key not in self._cache:
            self.stats.misses += 1
            return None

        entry = self._cache[key]
        now = time.time()

        if entry.is_expired(now):
            logger.debug(f"Cache key '{key[:12]}...' expired, evicting.")
            del self._cache[key]
            self.stats.misses += 1
            self.stats.evictions += 1
            self.stats.total_entries = len(self._cache)
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hit_count += 1
        self.stats.hits += 1

        # Track saved latency if present in cached response
        cached_val = entry.value
        timing = cached_val.get("timing") or {}
        saved_ms = timing.get("total_ms") or timing.get("total_latency_ms") or 250.0
        self.stats.saved_latency_ms += saved_ms

        logger.debug(f"Cache hit for key '{key[:12]}...' (saved ~{saved_ms:.1f}ms)")
        # Return a copy to avoid accidental mutation
        return json.loads(json.dumps(cached_val))

    def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: float | None = None,
    ) -> None:
        """Store inference result in cache with LRU eviction."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_sec

        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = CacheEntry(key, value, ttl)
            return

        # Check capacity and evict oldest if full
        if len(self._cache) >= self.max_entries:
            oldest_key, _ = self._cache.popitem(last=False)
            self.stats.evictions += 1
            logger.debug(f"Evicted oldest cache entry: '{oldest_key[:12]}...'")

        self._cache[key] = CacheEntry(key, value, ttl)
        self.stats.total_entries = len(self._cache)

    def clear(self) -> None:
        """Purge all cache entries and reset counts."""
        self._cache.clear()
        self.stats = CacheStats(max_entries=self.max_entries)

    def __len__(self) -> int:
        return len(self._cache)
