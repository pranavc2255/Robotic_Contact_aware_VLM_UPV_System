import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from upv_vlm_v2.pipeline.weight_prefetch import prefetch_during_perception


class PrefetchTests(unittest.TestCase):
    def test_reads_weights_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "weights").write_bytes(b"test" * 100)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "weights"}}))
            cfg = {"managed_qwen": {"enabled": True, "prefetch_weights": True, "model_path": tmp}}
            with patch("upv_vlm_v2.pipeline.weight_prefetch.available_memory", return_value=64 * 2**30):
                with prefetch_during_perception(cfg, root / "logs"):
                    pass
            result = json.loads((root / "logs/weight_prefetch.json").read_text())
            self.assertTrue(result["completed"])
            self.assertEqual(result["bytes_read"], 400)
            with patch("upv_vlm_v2.pipeline.weight_prefetch.available_memory", return_value=1):
                with prefetch_during_perception(cfg, root / "logs"):
                    pass
            result = json.loads((root / "logs/weight_prefetch.json").read_text())
            self.assertFalse(result["completed"])
            self.assertEqual(result["bytes_read"], 0)

    def test_dry_run_reads_nothing(self):
        with patch("upv_vlm_v2.pipeline.weight_prefetch.Path.read_text", side_effect=AssertionError):
            with prefetch_during_perception({"managed_qwen": {"enabled": True, "prefetch_weights": True}}, "/tmp/unused", dry_run=True):
                pass


if __name__ == "__main__":
    unittest.main()
