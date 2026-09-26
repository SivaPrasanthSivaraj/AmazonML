import tempfile
import unittest
from pathlib import Path

import polars as pl

from ber.pipeline_state import PipelineState
from ber.production_candidates import generate_classical_candidates


class ProductionCandidateTests(unittest.TestCase):
    def test_candidate_direction_and_country_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalized = root / "normalized"
            normalized.mkdir()
            common = {
                "business_name": ["Acme Shop", "Other"],
                "business_address": ["12 Main Street 560001", "99 Elsewhere"],
                "name_norm": ["acme shop", "other"],
                "address_norm": ["12 main street 560001", "99 elsewhere"],
                "name_compact": ["acmeshop", "other"],
                "address_compact": ["12mainstreet560001", "99elsewhere"],
                "source": ["S1", "S1"],
            }
            pl.DataFrame(
                {"entity_id": ["S1-1", "S1-2"], "country": ["US", "India"], **common}
            ).write_parquet(normalized / "test_source1.parquet")
            for source in (2, 3):
                pl.DataFrame(
                    {
                        "entity_id": [f"S{source}-1", f"S{source}-2"],
                        "country": ["US", "India"],
                        **{**common, "source": [f"S{source}", f"S{source}"]},
                    }
                ).write_parquet(normalized / f"test_source{source}.parquet")
            state = PipelineState(root / "work")
            result = generate_classical_candidates(
                normalized_dir=normalized,
                output_dir=root / "candidates",
                split="test",
                shard_count=2,
                top_k=5,
                state=state,
            )
            self.assertEqual(result["total_rows"], 4)
            pairs = pl.concat(
                [pl.read_parquet(path) for path in (root / "candidates" / "classical" / "test").rglob("part-*.parquet")]
            )
            self.assertEqual(
                set(pairs.select("s1_id", "target_id").iter_rows()),
                {("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-2", "S2-2"), ("S1-2", "S3-2")},
            )


if __name__ == "__main__":
    unittest.main()
