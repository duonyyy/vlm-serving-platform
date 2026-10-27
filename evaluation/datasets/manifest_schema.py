"""Benchmark dataset schema and validation."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class BenchmarkSample(BaseModel):
    """Schema for a benchmark test case."""

    sample_id: str
    image_path: str
    task_type: Literal["vqa", "ocr", "invoice"]
    category: str = Field(description="scene, document, chart, receipt, text")
    prompt: str
    ground_truth: Any = Field(description="Expected text string or structured dict")
    expected_observation: str | None = None
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    quality_tags: list[str] = Field(default_factory=list, description="photographed, low-light, skewed, clean")
