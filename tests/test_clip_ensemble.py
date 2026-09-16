import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference/src'))
from upv_vlm_v2.perception import clip_ensemble as ensemble


class ClipEnsembleTests(unittest.TestCase):
    def test_saved_candidate_ranking_all72(self):
        import csv
        grouped={}
        with (ROOT/'results/perception/candidate_scores.csv').open() as f:
            for r in csv.DictReader(f):
                if r['variant']=='object_surface_ensemble' and r['aggregation']==ensemble.MODE:
                    grouped.setdefault(r['trial'],[]).append(dict(candidate_id=r['candidate'],
                        clip_scores={m:float(r[m]) for m in ensemble.MATERIALS}))
        with (ROOT/'results/perception/trials72.csv').open() as f:
            for r in csv.DictReader(f):
                winner,selected=ensemble.choose(grouped[r['trial']],r['requested_material'],.35)
                self.assertEqual(winner['candidate_id'],r['candidate_id'])
                self.assertEqual(selected is not None,float(r['score'])>=.35)

    def row(self, identifier, brick, concrete, timber):
        return dict(candidate_id=identifier,clip_scores=dict(zip(ensemble.MATERIALS,[brick,concrete,timber])))

    def test_inclusive_threshold_and_no_margin_veto(self):
        row=self.row('A',.35,.60,.05)
        self.assertIs(ensemble.choose([row],'brick',.35)[1],row)
        self.assertIsNone(ensemble.choose([row],'brick',.350001)[1])
        self.assertEqual(ensemble.choose([],'brick',.35),(None,None))

    def test_requested_score_then_margin_tie(self):
        a=self.row('A',.4,.5,.1)
        b=self.row('B',.4,.3,.3)
        self.assertIs(ensemble.choose([a,b],'brick',.35)[1],b)
        b['clip_scores']['brick']=.39
        self.assertIs(ensemble.choose([a,b],'brick',.35)[1],a)

    def test_invalid_scores(self):
        for value in [float('nan'),float('inf'),-1,2]:
            with self.assertRaises(ValueError):
                ensemble.choose([self.row('A',value,.1,.1)],'brick',.35)

    def test_config_and_exact_prompts(self):
        cfg=json.loads((ROOT/'configs/pipeline.json').read_text())['target_selection']
        self.assertEqual(cfg['class_score_aggregation'],ensemble.MODE)
        self.assertEqual(cfg['clip_score_threshold'],.35)
        self.assertEqual(ensemble.prompts(),json.loads((ROOT/'results/perception/ensemble_prompts.json').read_text()))
        self.assertEqual([len(v) for v in ensemble.prompts().values()],[8,8,8])
        yaml=(ROOT/'configs/v2/deploy_ur3e_realsense_upv_real_plan_check_qwen32_multi_anchor_l6.yaml').read_text()
        self.assertIn('class_score_aggregation: '+ensemble.MODE,yaml)
        self.assertIn('clip_score_threshold: 0.35',yaml)

    def test_isolated_batch_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            args=dict(candidates=[dict(candidate_id='A')],requested_material='brick',
                config={'target_selection':{'clip_score_threshold':.35}},output_path=Path(temp)/'scores.json',
                crop_for_candidate=lambda c:('crop.png','inner_texture_crop'),
                python_info={'clip_subprocess_python':sys.executable})
            response=subprocess.CompletedProcess([],0,json.dumps({'scores':[dict(brick=.35,**{'concrete block':.6},timber=.05)]}), '')
            with patch.object(ensemble.subprocess,'run',return_value=response) as call:
                result=ensemble.verify(**args)
                self.assertEqual(call.call_count,1)
                self.assertEqual(result['selected_candidate_id'],'A')
                self.assertFalse(result['source_class_prior_used'])
                self.assertEqual(json.loads(call.call_args.kwargs['input'])['paths'],['crop.png'])
            with patch.object(ensemble.subprocess,'run',side_effect=RuntimeError('test failure')):
                result=ensemble.verify(**args)
                self.assertEqual(result['selected_candidate_id'],'NO_VERIFIED_MATCH')
                self.assertTrue(result['clip_verifier_failed'])
                self.assertFalse(result['target_absent_rejection_triggered'])
                self.assertFalse(result['no_verified_match'])


if __name__=='__main__':
    unittest.main()
