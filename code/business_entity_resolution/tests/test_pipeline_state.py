import tempfile
import unittest
from pathlib import Path

from ber.pipeline_state import PipelineState, atomic_write_json, signature


class PipelineStateTests(unittest.TestCase):
    def test_signature_is_stable_for_mapping_order(self):
        self.assertEqual(signature({"a": 1, "b": 2}), signature({"b": 2, "a": 1}))

    def test_stage_requires_signature_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = PipelineState(root)
            output = root / "result.parquet"
            state.begin_stage("features", "abc")
            output.write_bytes(b"ok")
            state.complete_stage("features", (output,))
            self.assertTrue(state.stage_complete("features", "abc", (output,)))
            self.assertFalse(state.stage_complete("features", "different", (output,)))
            output.unlink()
            self.assertFalse(state.stage_complete("features", "abc", (output,)))

    def test_shard_checkpoint_survives_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = PipelineState(root)
            state.begin_stage("candidates", "stage-signature")
            output = root / "part-000.parquet"
            output.write_bytes(b"ok")
            state.complete_shard(
                "candidates", "000", "shard-signature", (output,), {"rows": 4}
            )
            reloaded = PipelineState(root)
            self.assertTrue(
                reloaded.shard_complete(
                    "candidates", "000", "shard-signature", (output,)
                )
            )

    def test_atomic_json_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            atomic_write_json(path, {"ready": True})
            self.assertIn('"ready": true', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
