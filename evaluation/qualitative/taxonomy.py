"""Error Taxonomy for qualitative failure analysis in Vietnamese VLM Serving."""

from __future__ import annotations

import re
from typing import Any

ERROR_CATEGORIES = {
    "MISSING_TEXT": "Bỏ sót chữ hoặc nội dung quan trọng",
    "VIETNAMESE_DIACRITICS": "Sai dấu tiếng Việt hoặc nhầm đ/d",
    "CHARACTER_CONFUSION": "Nhầm ký tự tương đồng thị giác (0/O, 1/l, dấu câu)",
    "LAYOUT_ORDER": "Đọc sai thứ tự hàng/cột hoặc bố cục bảng",
    "AMOUNT_FIELD_CONFUSION": "Nhầm tổng tiền với VAT/tạm tính hoặc sai đơn vị tiền tệ",
    "HALLUCINATION": "Suy diễn thông tin không quan sát được trong ảnh",
    "RESOLUTION_LOSS": "Mất nét chữ nhỏ do resize hoặc giới hạn độ phân giải",
    "SCHEMA_FORMAT_ERROR": "Sai cú pháp JSON, thiếu trường bắt buộc hoặc lỗi runtime",
}

VIETNAMESE_DIACRITICS_REGEX = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.IGNORECASE)


def diagnose_sample_error(
    reference: str,
    prediction: str,
    status: str = "completed",
    task_type: str = "ocr",
) -> list[str]:
    """Classify the error types according to the formal project taxonomy.

    Returns:
        List of identified error taxonomy tags.
    """
    tags: list[str] = []

    if status != "completed" or not prediction.strip():
        tags.append("SCHEMA_FORMAT_ERROR")
        return tags

    ref_clean = reference.strip()
    pred_clean = prediction.strip()

    # 1. Check Missing Text
    if len(pred_clean) < 0.6 * len(ref_clean):
        tags.append("MISSING_TEXT")

    # 2. Check Vietnamese Diacritics
    # If reference contains Vietnamese diacritics but prediction has stripped or modified them
    ref_diacritics = len(VIETNAMESE_DIACRITICS_REGEX.findall(ref_clean))
    pred_diacritics = len(VIETNAMESE_DIACRITICS_REGEX.findall(pred_clean))
    if ref_diacritics > 3 and pred_diacritics < ref_diacritics * 0.7:
        tags.append("VIETNAMESE_DIACRITICS")

    # 3. Check Character Confusion: numbers 0 vs O, 1 vs l
    if re.search(r"\b0\b", ref_clean) and re.search(r"\bO\b", pred_clean, re.IGNORECASE):
        tags.append("CHARACTER_CONFUSION")
    elif re.search(r"\b1\b", ref_clean) and re.search(r"\bl\b", pred_clean):
        tags.append("CHARACTER_CONFUSION")

    # 4. Check Amount / Currency confusion in invoice tasks
    if task_type == "invoice":
        if ("tổng" in ref_clean.lower() or "total" in ref_clean.lower()) and ("vat" in pred_clean.lower() or "thuế" in pred_clean.lower()):
            tags.append("AMOUNT_FIELD_CONFUSION")

    # 5. Check Hallucination (prediction much longer than reference with fabricated text)
    if len(pred_clean) > 2.0 * len(ref_clean) and len(ref_clean) > 20:
        tags.append("HALLUCINATION")

    # Default to general missing text if there's an edit error without specific tag
    if not tags and ref_clean != pred_clean:
        tags.append("MISSING_TEXT")

    return tags
