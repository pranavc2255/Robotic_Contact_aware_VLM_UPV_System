import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from upv_vlm_contact.cli import contact, paths, perception, runtime
from upv_vlm_contact.data import safe_member
from upv_vlm_contact.experiments import parse_response
from upv_vlm_contact.metrics import boolean, classify, select_usable


class ReproductionTests(unittest.TestCase):
    def test_no_model_import(self):
        for name in ['torch', 'transformers', 'cv2', 'pyrealsense2', 'rtde_control']:
            self.assertNotIn(name, sys.modules)

    def test_contact_golden_counts(self):
        expected = {
            'contact73': {'Qwen2.5-32B': (43, 6, 16, 8, 15), 'Qwen3-2B': (41, 8, 14, 10, 13),
                'Qwen2.5-3B': (36, 13, 9, 15, 13), 'Classical RGB+mask': (42, 12, 10, 9, 13),
                'Classical RGB-D': (34, 8, 14, 17, 13)},
            'contact82': {'Qwen2.5-32B': (44, 9, 21, 8, 17), 'Qwen3-2B': (43, 13, 17, 9, 16),
                'Qwen2.5-3B': (38, 14, 16, 14, 16), 'Classical RGB+mask': (42, 10, 20, 10, 15),
                'Classical RGB-D': (37, 6, 24, 15, 16)}}
        for dataset, models in expected.items():
            result = contact(ROOT, dataset)['metrics']
            self.assertEqual(set(result), set(models))
            for model, counts in models.items():
                self.assertEqual(tuple(result[model][f] for f in ['TP','FP','TN','FN','selected_good']), counts)

    def test_perception(self):
        result = perception(ROOT)
        self.assertEqual(result['correct'], 57)
        self.assertEqual(result['original27']['correct'], 27)
        self.assertEqual(result['revision36']['correct'], 30)

    def test_path_summaries(self):
        for version in paths(ROOT).values():
            self.assertEqual(version['all']['mask']['n_valid'], 30)
            self.assertEqual(version['all']['depth_point_cloud']['n_valid'], 30)

    def test_runtime_rows(self):
        result = runtime(ROOT)['recorded_successful_rows']['total_pipeline_timing_ms']
        self.assertEqual(result['n'], 94)
        self.assertTrue(math.isclose(result['mean_s'], 98.87749526595745))

    def test_labels_and_ties(self):
        with self.assertRaises(ValueError):
            boolean('unknown')
        self.assertIsNone(classify([])['precision'])
        rows = [dict(anchor_id='A2', predicted_good=True, score=0),
                dict(anchor_id='A1', predicted_good=True, score=0)]
        self.assertEqual(select_usable(rows)['anchor_id'], 'A1')
        self.assertIsNone(select_usable([dict(predicted_good=False)]))

    def test_response_schema(self):
        obj = dict(per_anchor_analysis=[dict(anchor_id='A1', anchor_usable=True, anchor_score=0)])
        self.assertEqual(parse_response(dict(text=json.dumps(obj)), ['A1']), obj)
        with self.assertRaises(ValueError):
            parse_response(dict(text=json.dumps(obj)), ['A1', 'A2'])
        obj['per_anchor_analysis'][0]['anchor_usable'] = 'true'
        with self.assertRaises(ValueError):
            parse_response(dict(text=json.dumps(obj)), ['A1'])

    def test_archive_paths(self):
        self.assertTrue(safe_member('objects/123.png'))
        for path in ['../bad', '/etc/passwd', 'C:/bad', 'a/../bad', 'a//b', 'a/./b']:
            self.assertFalse(safe_member(path))

    def test_input_cohorts(self):
        rows = json.loads((ROOT/'data/manifests/contact_inputs.json').read_text())
        for dataset, count in [('contact73', 73), ('contact82', 82)]:
            ids = [(r['split'], r['case'], r['anchor_id']) for r in rows if r['dataset'] == dataset]
            self.assertEqual(len(ids), count)
            self.assertEqual(len(set(ids)), count)


if __name__ == '__main__':
    unittest.main()
