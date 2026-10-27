"""Quantitative metrics: CER, WER, Exact Match, ANLS, Field F1, JSON validity."""

from __future__ import annotations

import json
import re
from typing import Any

from .normalizer import normalize_currency, normalize_date, normalize_text


def levenshtein_distance(a: list[str] | str, b: list[str] | str) -> int:
    """Compute Levenshtein distance between two sequences with pure Python fallback."""
    try:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(a, b))
    except ImportError:
        # Dynamic programming matrix (memory optimized to 2 rows)
        previous = list(range(len(b) + 1))
        for i, left in enumerate(a, start=1):
            current = [i]
            for j, right in enumerate(b, start=1):
                cost = 0 if left == right else 1
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
            previous = current
        return previous[-1]


def compute_cer(reference: str, hypothesis: str, mode: str = "normalized") -> float:
    """Character Error Rate: edit_distance(ref, hyp) / len(ref). Returns 0.0 if both empty."""
    ref_norm = normalize_text(reference, mode=mode)
    hyp_norm = normalize_text(hypothesis, mode=mode)

    if not ref_norm and not hyp_norm:
        return 0.0
    if not ref_norm:
        return 1.0

    dist = levenshtein_distance(list(ref_norm), list(hyp_norm))
    return round(dist / len(ref_norm), 4)


def compute_wer(reference: str, hypothesis: str, mode: str = "normalized") -> float:
    """Word Error Rate: word_edit_distance(ref, hyp) / len(ref_words). Returns 0.0 if both empty."""
    ref_norm = normalize_text(reference, mode=mode)
    hyp_norm = normalize_text(hypothesis, mode=mode)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return round(dist / len(ref_words), 4)


def compute_exact_match(reference: str, hypothesis: str, mode: str = "normalized") -> bool:
    """Exact Match boolean check."""
    ref_norm = normalize_text(reference, mode=mode)
    hyp_norm = normalize_text(hypothesis, mode=mode)
    return ref_norm == hyp_norm


def compute_anls(reference: str, hypothesis: str, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity for VQA tasks.

    Formula: 1 - min(1, dist / max(len(ref), len(hyp))) if <= (1 - threshold), else 0.
    """
    ref_norm = normalize_text(reference, mode="normalized").lower()
    hyp_norm = normalize_text(hypothesis, mode="normalized").lower()

    if not ref_norm and not hyp_norm:
        return 1.0
    if not ref_norm or not hyp_norm:
        return 0.0

    max_len = max(len(ref_norm), len(hyp_norm))
    dist = levenshtein_distance(list(ref_norm), list(hyp_norm))
    norm_dist = dist / max_len

    if norm_dist >= threshold:
        return 0.0
    return round(1.0 - norm_dist, 4)


def compute_field_f1(gold_data: dict[str, Any], pred_data: dict[str, Any]) -> dict[str, Any]:
    """Calculate Precision, Recall, and F1 across structured document fields.

    Handles monetary and date normalization before equality checking.
    """
    field_results: dict[str, dict[str, Any]] = {}
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    all_keys = sorted(set(gold_data.keys()) | set(pred_data.keys()))

    for key in all_keys:
        gold_val = gold_data.get(key)
        pred_val = pred_data.get(key)

        # Normalize special types
        if "amount" in key.lower() or "price" in key.lower() or "total" in key.lower():
            g_comp = normalize_currency(gold_val)
            p_comp = normalize_currency(pred_val)
        elif "date" in key.lower():
            g_comp = normalize_date(gold_val)
            p_comp = normalize_date(pred_val)
        else:
            g_comp = normalize_text(str(gold_val), mode="normalized").lower() if gold_val is not None else None
            p_comp = normalize_text(str(pred_val), mode="normalized").lower() if pred_val is not None else None

        # Check match
        if g_comp is not None and p_comp is not None:
            if g_comp == p_comp:
                true_positives += 1
                match = True
            else:
                false_positives += 1
                false_negatives += 1
                match = False
        elif g_comp is not None and p_comp is None:
            false_negatives += 1
            match = False
        elif g_comp is None and p_comp is not None:
            false_positives += 1
            match = False
        else:
            # Both None: agreement on null value (abstain)
            true_positives += 1
            match = True

        field_results[key] = {
            "gold": gold_val,
            "pred": pred_val,
            "match": match,
        }

    precision = round(true_positives / (true_positives + false_positives), 4) if (true_positives + false_positives) > 0 else 0.0
    recall = round(true_positives / (true_positives + false_negatives), 4) if (true_positives + false_negatives) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fields": field_results,
    }


def compute_json_validity(raw_text: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Test if prediction contains parseable JSON."""
    if not raw_text:
        return False, None, "empty output"

    clean = raw_text.strip()
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return True, data, None
        return False, None, f"JSON parsed into non-dict: {type(data).__name__}"
    except Exception as exc:
        return False, None, str(exc)
