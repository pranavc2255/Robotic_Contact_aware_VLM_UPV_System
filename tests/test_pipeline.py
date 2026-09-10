"""Pipeline planning tests: standard library only, no model or hardware imports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from upv_vlm_contact.pipeline import STAGES, file_record, plan


class PipelinePlanningTests(unittest.TestCase):
    def test_all_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(plan(list(STAGES), Path(tmp)), list(STAGES))

    def test_dependencies_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            for stages in [['paths'], ['contact'], ['anchors','perception'], ['perception','perception']]:
                with self.assertRaises(ValueError):
                    plan(stages, Path(tmp))

    def test_resume_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'perception').mkdir()
            (folder/'perception/result.json').write_text(json.dumps({'success':True}))
            self.assertEqual(plan(['anchors'], folder), ['anchors'])
            with self.assertRaises(FileExistsError):
                plan(['perception'], folder)
            (folder/'perception/result.json').write_text(json.dumps({'success':False}))
            with self.assertRaises(ValueError):
                plan(['anchors'], folder)

    def test_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'input'
            file.write_bytes(b'one')
            first=file_record(file)
            file.write_bytes(b'two')
            self.assertNotEqual(first, file_record(file))


if __name__ == '__main__':
    unittest.main()
