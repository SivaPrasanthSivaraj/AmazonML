import unittest

from ber.normalization import normalize_text


class NormalizationTests(unittest.TestCase):
    def test_punctuation_case_and_spacing(self):
        self.assertEqual(normalize_text("  Payne--ENTERPRISES, LLC "), "payne enterprises llc")

    def test_unicode_is_preserved(self):
        self.assertEqual(normalize_text("एसएस फ़ूड"), "एसएस फ़ूड")

    def test_optional_accent_fold(self):
        self.assertEqual(normalize_text("École", strip_marks=True), "ecole")

    def test_null(self):
        self.assertEqual(normalize_text(None), "")


if __name__ == "__main__":
    unittest.main()
