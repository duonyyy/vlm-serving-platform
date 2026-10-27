"""Text, currency, and date normalization for Vietnamese VLM evaluation."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Regex to match Vietnamese currency amounts like "150.000 đ", "150,000 VND", "150000"
CURRENCY_PATTERN = re.compile(
    r"([\d\.,\s]+)\s*(?:vnđ|vnd|đồng|đ|d)?",
    re.IGNORECASE,
)

# Common date patterns: YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY
DATE_DMY_PATTERN = re.compile(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$")
DATE_YMD_PATTERN = re.compile(r"^(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})$")


def normalize_text(text: str, mode: str = "normalized") -> str:
    """Normalize text with Unicode NFC and whitespace standardization.

    Args:
        text: Input string.
        mode: "strict" (no change) or "normalized" (Unicode NFC, strip extra whitespace).
    """
    if not text:
        return ""
    if mode == "strict":
        return text
    # Unicode NFC normalization is vital for Vietnamese (chữ có dấu như ệ, ờ, ắ)
    normalized = unicodedata.normalize("NFC", text)
    # Collapse multiple whitespace, tabs, and newlines into single spaces
    return " ".join(normalized.replace("\r\n", " ").replace("\n", " ").split()).strip()


def normalize_currency(val: Any) -> float | None:
    """Normalize various currency expressions to a float value.

    Examples:
        - "150.000 VND" -> 150000.0
        - "1,250,000.50" -> 1250000.5
        - 150000 -> 150000.0
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    text = str(val).strip().lower()
    if not text:
        return None

    # Remove currency words
    text = re.sub(r"[^\d\.,]", "", text)
    if not text:
        return None

    # Determine thousand vs decimal separator
    if "." in text and "," in text:
        if text.rfind(".") > text.rfind(","):
            # Format: 1,250.50
            text = text.replace(",", "")
        else:
            # Format: 1.250,50
            text = text.replace(".", "").replace(",", ".")
    elif "." in text:
        # If dot is used as thousand separator (e.g. 150.000)
        parts = text.split(".")
        if len(parts[-1]) == 3 or len(parts) > 2:
            text = text.replace(".", "")
        else:
            # Could be a decimal (e.g. 15.5)
            pass
    elif "," in text:
        parts = text.split(",")
        if len(parts[-1]) == 3 or len(parts) > 2:
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def normalize_date(val: Any) -> str | None:
    """Normalize date strings into ISO format YYYY-MM-DD.

    Examples:
        - "13/10/2026" -> "2026-10-13"
        - "2026-10-13" -> "2026-10-13"
    """
    if not val:
        return None
    text = str(val).strip()

    match_ymd = DATE_YMD_PATTERN.match(text)
    if match_ymd:
        year, month, day = match_ymd.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    match_dmy = DATE_DMY_PATTERN.match(text)
    if match_dmy:
        day, month, year = match_dmy.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    return text
