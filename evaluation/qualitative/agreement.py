"""Human qualitative review rubric and inter-annotator agreement metrics."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class QualitativeReviewItem(BaseModel):
    """Rubric for manual evaluation according to Section 8.2 of KEHOACH.md."""

    sample_id: str
    reviewer: str
    grounded_in_image: bool = Field(description="1. Đúng nội dung nhìn thấy trong ảnh")
    complete_info: bool = Field(description="2. Đầy đủ thông tin được yêu cầu")
    no_hallucination: bool = Field(description="3. Không suy diễn giá trị không quan sát được")
    vietnamese_clarity: bool = Field(description="4. Tiếng Việt rõ ràng, đúng dấu")
    format_adherence: bool = Field(description="5. Làm đúng định dạng Markdown/JSON yêu cầu")
    overall_verdict: str = Field(description="'PASS', 'PARTIAL', or 'FAIL'")
    notes: str | None = None


def compute_cohens_kappa(rater1: list[str], rater2: list[str]) -> float:
    """Calculate Cohen's Kappa score for inter-rater agreement on categorical verdicts."""
    if len(rater1) != len(rater2) or not rater1:
        return 0.0

    n = len(rater1)
    categories = sorted(set(rater1) | set(rater2))

    # Observed agreement Po
    matches = sum(1 for a, b in zip(rater1, rater2) if a == b)
    po = matches / n

    # Expected agreement Pe
    count1 = {c: rater1.count(c) for c in categories}
    count2 = {c: rater2.count(c) for c in categories}
    pe = sum((count1[c] / n) * (count2[c] / n) for c in categories)

    if pe == 1.0:
        return 1.0
    kappa = (po - pe) / (1.0 - pe)
    return round(kappa, 4)
