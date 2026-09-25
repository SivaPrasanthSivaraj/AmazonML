import unittest

from ber.metrics import entity_fbeta, macro_fbeta, parse_id_list


class ParseIdListTests(unittest.TestCase):
    def test_empty_values(self):
        self.assertEqual(parse_id_list(None), frozenset())
        self.assertEqual(parse_id_list(""), frozenset())

    def test_ids_are_sets(self):
        self.assertEqual(parse_id_list("S2-1,S3-2,S2-1"), {"S2-1", "S3-2"})


class FbetaTests(unittest.TestCase):
    def test_singleton_rules(self):
        self.assertEqual(entity_fbeta([], []), 1.0)
        self.assertEqual(entity_fbeta([], ["S2-1"]), 0.0)
        self.assertEqual(entity_fbeta(["S2-1"], []), 0.0)

    def test_problem_statement_example(self):
        score = entity_fbeta(
            ["S2-00047", "S3-00812"],
            ["S2-00047", "S2-00193", "S3-00812"],
        )
        self.assertAlmostEqual(score, 5 / 7)

    def test_macro_average_includes_singletons(self):
        truth = {"S1-1": set(), "S1-2": {"S2-1"}}
        prediction = {"S1-1": set(), "S1-2": set()}
        self.assertEqual(macro_fbeta(truth, prediction), 0.5)

    def test_macro_requires_exact_s1_coverage(self):
        with self.assertRaises(ValueError):
            macro_fbeta({"S1-1": set()}, {})


if __name__ == "__main__":
    unittest.main()
