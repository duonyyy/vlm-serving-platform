"""Unit tests for Phase 3 evaluation quality metrics and normalizers."""

import unittest
from evaluation.quality.metrics import (
    compute_anls,
    compute_cer,
    compute_exact_match,
    compute_field_f1,
    compute_json_validity,
    compute_wer,
    levenshtein_distance,
)
from evaluation.quality.normalizer import (
    normalize_currency,
    normalize_date,
    normalize_text,
)


class EvaluationQualityTests(unittest.TestCase):
    def test_levenshtein_distance(self) -> None:
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance(["a", "b"], ["a", "b"]), 0)
        self.assertEqual(levenshtein_distance(["a", "b"], ["a", "c"]), 1)

    def test_cer_and_wer(self) -> None:
        # Exact match
        self.assertEqual(compute_cer("Hà Nội", "Hà Nội"), 0.0)
        self.assertEqual(compute_wer("Hà Nội", "Hà Nội"), 0.0)

        # Diacritic difference
        cer_diacritic = compute_cer("Hà Nội", "Ha Noi")
        self.assertGreater(cer_diacritic, 0.0)

        # Empty reference handling
        self.assertEqual(compute_cer("", ""), 0.0)
        self.assertEqual(compute_cer("", "text"), 1.0)

    def test_exact_match(self) -> None:
        self.assertTrue(compute_exact_match("Cà phê   Việt", "Cà phê Việt"))
        self.assertFalse(compute_exact_match("Cà phê Việt", "Ca phe Viet"))

    def test_anls_metric(self) -> None:
        self.assertEqual(compute_anls("Màu xanh", "màu xanh"), 1.0)
        self.assertEqual(compute_anls("Màu xanh", "Màu đỏ"), 0.0)  # > 0.5 edit dist

    def test_currency_normalization(self) -> None:
        self.assertEqual(normalize_currency("150.000 đ"), 150000.0)
        self.assertEqual(normalize_currency("150,000 VND"), 150000.0)
        self.assertEqual(normalize_currency("1.250.000 VNĐ"), 1250000.0)
        self.assertEqual(normalize_currency("12.50"), 12.5)
        self.assertEqual(normalize_currency(50000), 50000.0)
        self.assertIsNone(normalize_currency(None))

    def test_date_normalization(self) -> None:
        self.assertEqual(normalize_date("13/10/2026"), "2026-10-13")
        self.assertEqual(normalize_date("2026-10-13"), "2026-10-13")
        self.assertEqual(normalize_date("13-10-2026"), "2026-10-13")

    def test_field_f1_calculation(self) -> None:
        gold = {
            "seller_name": "Công ty ABC",
            "total_amount": 150000.0,
            "invoice_date": "2026-10-13",
            "tax_amount": None,
        }
        # Prediction with formatted currency and date
        pred = {
            "seller_name": "công ty abc",
            "total_amount": "150.000 VND",
            "invoice_date": "13/10/2026",
            "tax_amount": None,
        }
        res = compute_field_f1(gold, pred)
        self.assertEqual(res["f1"], 1.0)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)

        # Mismatched amount
        pred_wrong = dict(pred, total_amount="200.000 VND")
        res_wrong = compute_field_f1(gold, pred_wrong)
        self.assertLess(res_wrong["f1"], 1.0)

    def test_json_validity(self) -> None:
        valid_raw = "```json\n{\"total\": 100}\n```"
        is_valid, parsed, err = compute_json_validity(valid_raw)
        self.assertTrue(is_valid)
        self.assertEqual(parsed, {"total": 100})

        invalid_raw = "Đây là kết quả: không phải json"
        is_valid2, _, _ = compute_json_validity(invalid_raw)
        self.assertFalse(is_valid2)


if __name__ == "__main__":
    unittest.main()
