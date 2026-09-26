import tempfile
import unittest
from pathlib import Path

import polars as pl

from ber.pipeline_state import PipelineState
from ber.production_data import prepare_production_data


class ProductionDataTests(unittest.TestCase):
    def _write_source(self, path: Path, prefix: str):
        pl.DataFrame(
            {
                "entity_id": [f"{prefix}-1"],
                "business_name": ["Caf\u00e9, LLC"],
                "business_address": ["12 Main St."],
                "country": ["US"],
            }
        ).write_csv(path, separator="\t")

    def test_prepare_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            for split in ("train", "test"):
                (dataset / split).mkdir(parents=True)
                for source in (1, 2, 3):
                    self._write_source(
                        dataset / split / f"{split}_source{source}.tsv",
                        f"S{source}",
                    )
            pl.DataFrame(
                {
                    "source1_entity_id": ["S1-1"],
                    "matched_entity_ids": ["S2-1,S3-1"],
                }
            ).write_csv(dataset / "train" / "train_ground_truth.tsv", separator="\t")
            work = root / "work"
            state = PipelineState(work)
            first = prepare_production_data(dataset, work / "normalized", state)
            second = prepare_production_data(dataset, work / "normalized", state)
            self.assertFalse(first["skipped"])
            self.assertTrue(second["skipped"])
            source = pl.read_parquet(work / "normalized" / "train_source1.parquet")
            self.assertEqual(source["name_norm"].item(), "cafe llc")
            truth = pl.read_parquet(work / "normalized" / "train_truth_links.parquet")
            self.assertEqual(truth.height, 2)


if __name__ == "__main__":
    unittest.main()
