import unittest

import polars as pl

from ber.ann_retrieval import _marginal_summary, _recall_rows, _text_for_view


class AnnTextTests(unittest.TestCase):
    def test_symmetric_e5_prefix_and_views(self):
        self.assertEqual(_text_for_view("Acme", "Road", "name"), "query: Acme")
        self.assertEqual(_text_for_view("Acme", "Road", "address"), "query: Road")
        self.assertIn(
            "business name: Acme; business address: Road",
            _text_for_view("Acme", "Road", "combined"),
        )


class AnnEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.truth = pl.DataFrame(
            {
                "s1_id": ["a", "b", "c"],
                "target_id": ["x", "y", "z"],
                "country": ["India", "India", "US"],
            }
        )
        self.candidates = pl.DataFrame(
            {
                "s1_id": ["a", "q", "q", "b"],
                "target_id": ["x", "x", "y", "y"],
                "ann_rank": [1, 2, 1, 2],
                "view": ["name", "name", "name", "address"],
            }
        )

    def test_union_recall_at_k(self):
        rows = _recall_rows(self.truth, self.candidates, (1, 2))
        self.assertEqual(rows[0]["retrieved"], 1)
        self.assertEqual(rows[1]["retrieved"], 2)
        self.assertAlmostEqual(rows[1]["recall"], 2 / 3)

    def test_ann_marginal_over_classical(self):
        classical = pl.DataFrame({"s1_id": ["a"], "target_id": ["x"]})
        result = _marginal_summary(
            self.truth, self.candidates, classical, top_k=2
        )
        self.assertEqual(result["classical_retrieved"], 1)
        self.assertEqual(result["ann_retrieved"], 2)
        self.assertEqual(result["ann_marginal_links"], 1)
        self.assertEqual(result["union_retrieved"], 2)


if __name__ == "__main__":
    unittest.main()
