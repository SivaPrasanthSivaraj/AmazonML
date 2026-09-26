import unittest

import numpy as np

from ber.production_ann import _production_candidate_frame, _shard_mask


class ProductionAnnTests(unittest.TestCase):
    def test_candidates_use_s1_to_target_direction(self):
        frame = _production_candidate_frame(
            s1_ids=["S1-1", "S1-2"],
            target_ids=["S2-1", "S2-2", "S2-3"],
            scores=np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
            indices=np.asarray([[2, 0], [1, 2]], dtype=np.int64),
            source=2,
            country="US",
            view="combined",
        )
        self.assertEqual(
            frame.select("s1_id", "target_id").rows(),
            [
                ("S1-1", "S2-3"),
                ("S1-1", "S2-1"),
                ("S1-2", "S2-2"),
                ("S1-2", "S2-3"),
            ],
        )

    def test_numeric_shards_are_disjoint(self):
        ids = ["S1-1", "S1-2", "S1-3", "S1-4"]
        self.assertEqual(_shard_mask(ids, 0, 2).tolist(), [False, True, False, True])
        self.assertEqual(_shard_mask(ids, 1, 2).tolist(), [True, False, True, False])


if __name__ == "__main__":
    unittest.main()
