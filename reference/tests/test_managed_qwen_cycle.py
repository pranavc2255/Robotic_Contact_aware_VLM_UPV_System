"""Offline lifecycle and selection tests: no model, network or hardware calls."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from upv_vlm_v2.pipeline.managed_qwen import managed_qwen
from upv_vlm_v2.anchor_selection.qwen32_multi_anchor_l6_batch_ranker import _strict_score_rank


class ManagedQwenTests(unittest.TestCase):
    def config(self):
        return {"managed_qwen": {"enabled": True, "python": "python", "launcher": "server.py", "model_path": "/tmp/model"},
                "anchor_selection": {"qwen_server_url": "http://127.0.0.1:8896"}}

    def test_dry_run_does_not_start_server(self):
        with patch("upv_vlm_v2.pipeline.managed_qwen.subprocess.Popen") as spawn:
            with managed_qwen(self.config(), "/tmp/unused", dry_run=True):
                pass
            spawn.assert_not_called()

    def test_stops_owned_server_on_success_and_failure(self):
        for failure in (False, True):
            with tempfile.TemporaryDirectory() as tmp, \
                 patch("upv_vlm_v2.pipeline.managed_qwen.validate_managed_qwen"), \
                 patch("upv_vlm_v2.pipeline.managed_qwen.subprocess.Popen") as spawn, \
                 patch("upv_vlm_v2.pipeline.managed_qwen.urlopen") as health:
                spawn.return_value.poll.return_value = None
                spawn.return_value.pid = 123
                health.return_value.__enter__.return_value = io.StringIO(json.dumps({"model_loaded": True, "model_path": "/tmp/model"}))
                try:
                    with managed_qwen(self.config(), tmp):
                        if failure:
                            raise ValueError("inference failed")
                except ValueError:
                    self.assertTrue(failure)
                spawn.return_value.terminate.assert_called_once()
                spawn.return_value.wait.assert_called_once()
                record = json.loads((Path(tmp) / "managed_qwen_lifecycle.json").read_text())
                self.assertEqual(record["success"], not failure)

    def test_server_early_exit_aborts(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch("upv_vlm_v2.pipeline.managed_qwen.validate_managed_qwen"), \
             patch("upv_vlm_v2.pipeline.managed_qwen.subprocess.Popen") as spawn:
            spawn.return_value.poll.return_value = 1
            spawn.return_value.pid = 123
            with self.assertRaises(RuntimeError):
                with managed_qwen(self.config(), tmp):
                    self.fail("Must not reach inference/motion")

    def test_selection_uses_scores_not_model_ranking(self):
        payload = {"ranked_usable_anchors": ["A2", "A1"], "per_anchor_analysis": [
            {"anchor_id": "A1", "anchor_usable": True, "anchor_score": 90},
            {"anchor_id": "A2", "anchor_usable": True, "anchor_score": 80}]}
        self.assertEqual(_strict_score_rank(payload, ["A1", "A2"]), ["A1", "A2"])
        for row in payload["per_anchor_analysis"]:
            row["anchor_usable"] = False
        self.assertEqual(_strict_score_rank(payload, ["A1", "A2"]), [])
        payload["per_anchor_analysis"].pop()
        with self.assertRaises(ValueError):
            _strict_score_rank(payload, ["A1", "A2"])


if __name__ == "__main__":
    unittest.main()
