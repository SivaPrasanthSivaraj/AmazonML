import tempfile
import unittest
from pathlib import Path

import polars as pl

from ber.pipeline_state import PipelineState
from ber.production_pair_union import build_pair_shards


class ProductionPairUnionTests(unittest.TestCase):
    def test_truth_is_forced_only_for_fitting_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalized = root / "normalized"
            normalized.mkdir()
            pl.DataFrame(
                {
                    "entity_id": ["S1-1", "S1-2"],
                    "country": ["US", "US"],
                    "split": ["train", "calibration"],
                }
            ).write_parquet(normalized / "train_source1.parquet")
            pl.DataFrame(
                {
                    "s1_id": ["S1-1", "S1-2"],
                    "target_id": ["S2-9", "S2-8"],
                    "source": ["S2", "S2"],
                }
            ).write_parquet(normalized / "train_truth_links.parquet")

            candidates = root / "candidates"
            for source in (2, 3):
                classical_path = candidates / "classical" / "train" / f"S{source}" / "part-0000.parquet"
                classical_path.parent.mkdir(parents=True, exist_ok=True)
                pl.DataFrame(
                    {
                        "s1_id": ["S1-1", "S1-2"],
                        "target_id": [f"S{source}-1", f"S{source}-2"],
                        "country": ["US", "US"],
                        "source": [f"S{source}", f"S{source}"],
                        "classical_evidence": [1.0, 1.0],
                    }
                ).write_parquet(classical_path)
                ann_path = candidates / "ann" / "train" / f"S{source}" / "combined" / "us" / "part-0000.parquet"
                ann_path.parent.mkdir(parents=True, exist_ok=True)
                pl.DataFrame(
                    {
                        "s1_id": ["S1-1", "S1-2"],
                        "target_id": [f"S{source}-1", f"S{source}-2"],
                        "source": [f"S{source}", f"S{source}"],
                        "country": ["US", "US"],
                        "ann_view": ["combined", "combined"],
                        "ann_rank": [1, 1],
                        "ann_score": [0.8, 0.8],
                    }
                ).write_parquet(ann_path)

            output = root / "pairs"
            build_pair_shards(
                normalized_dir=normalized,
                candidates_dir=candidates,
                output_dir=output,
                split="train",
                state=PipelineState(root / "state"),
                shard_count=1,
                train_negative_cap_per_s1_source=2,
            )
            s2 = pl.read_parquet(output / "pairs" / "train" / "S2" / "part-0000.parquet")
            pair_set = set(s2.select("s1_id", "target_id").iter_rows())
            self.assertIn(("S1-1", "S2-9"), pair_set)
            self.assertNotIn(("S1-2", "S2-8"), pair_set)


if __name__ == "__main__":
    unittest.main()
