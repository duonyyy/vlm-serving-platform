"""Unit tests for qualitative error taxonomy and inter-rater agreement."""

import unittest
from evaluation.qualitative.agreement import compute_cohens_kappa
from evaluation.qualitative.taxonomy import diagnose_sample_error


class EvaluationQualitativeTests(unittest.TestCase):
    def test_diagnose_sample_error(self) -> None:
        # 1. Missing text diagnosis
        tags1 = diagnose_sample_error(reference="Một đoạn văn rất dài đầy đủ các ý chính", prediction="Một đoạn")
        self.assertIn("MISSING_TEXT", tags1)

        # 2. Vietnamese diacritics loss
        tags2 = diagnose_sample_error(
            reference="Cộng hòa xã hội chủ nghĩa Việt Nam độc lập tự do hạnh phúc",
            prediction="Cong hoa xa hoi chu nghia Viet Nam doc lap tu do hanh phuc",
        )
        self.assertIn("VIETNAMESE_DIACRITICS", tags2)

        # 3. Failed run diagnosis
        tags3 = diagnose_sample_error(reference="Text", prediction="", status="failed")
        self.assertIn("SCHEMA_FORMAT_ERROR", tags3)

    def test_compute_cohens_kappa(self) -> None:
        # Perfect agreement
        r1 = ["PASS", "PASS", "FAIL", "PARTIAL", "PASS"]
        r2 = ["PASS", "PASS", "FAIL", "PARTIAL", "PASS"]
        self.assertEqual(compute_cohens_kappa(r1, r2), 1.0)

        # Disagreement
        r3 = ["FAIL", "FAIL", "PASS", "PASS", "FAIL"]
        kappa = compute_cohens_kappa(r1, r3)
        self.assertLess(kappa, 0.5)


if __name__ == "__main__":
    unittest.main()
