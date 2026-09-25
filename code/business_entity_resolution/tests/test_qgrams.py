import unittest

import polars as pl

from ber.qgram_retrieval import qgrams_expr


class QgramTests(unittest.TestCase):
    def test_overlapping_qgrams(self):
        frame = pl.DataFrame({"text": ["ABCDEFG"]})
        grams = set(frame.select(qgrams_expr("text", 4)).item())
        self.assertEqual(grams, {"abcd", "bcde", "cdef", "defg"})

    def test_short_text_has_no_qgrams(self):
        frame = pl.DataFrame({"text": ["abc"]})
        self.assertEqual(frame.select(qgrams_expr("text", 4)).item().to_list(), [])


if __name__ == "__main__":
    unittest.main()
