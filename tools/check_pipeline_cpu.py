"""Synthetic CPU smoke test of saved-mask -> anchors -> mocked decision -> paths.

Not a benchmark. No perception/VLM models or network services are used.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import io

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from upv_vlm_contact.pipeline import worker


def main():
    with tempfile.TemporaryDirectory(prefix='upv_pipeline_smoke_') as tmp:
        temp=Path(tmp)
        mask=np.zeros((240,400),np.uint8)
        mask[60:180,40:360]=255
        Image.fromarray(mask).save(temp/'mask.png')
        Image.fromarray(np.repeat(mask[:,:,None],3,axis=2)).save(temp/'rgb.png')
        np.save(temp/'depth.npy',np.full(mask.shape,300,np.uint16))
        (temp/'camera.json').write_text(json.dumps({'fx':300.,'fy':300.,'cx':200.,'cy':120.}))
        out=temp/'run'
        env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MPLBACKEND':'Agg'}
        base=[sys.executable,'-m','upv_vlm_contact.pipeline','--root',str(ROOT),'--run-dir',str(out)]
        subprocess.run(base+['--rgb',str(temp/'rgb.png'),'--depth',str(temp/'depth.npy'),
             '--camera-info',str(temp/'camera.json'),'--mask',str(temp/'mask.png'),
             '--material','concrete block','--stages','perception','anchors','--run'],cwd=ROOT,env=env,check=True)
        anchors=json.loads((out/'anchors/result.json').read_text())
        assert anchors['success'] and len(anchors['images'])>0
        (out/'contact').mkdir()
        response={'ok':True,'text':json.dumps({'per_anchor_analysis':[
            {'anchor_id':aid,'anchor_usable':True,'anchor_score':80}
            for aid in anchors['images']]})}
        with patch('upv_vlm_contact.pipeline.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as mocked:
            worker(ROOT, out, 'contact')
            assert mocked.call_args.args[0].full_url.endswith('/infer_multi')
        contact=json.loads((out/'contact/result.json').read_text())
        assert contact['selected_anchor_id']=='A1'
        request=json.loads((out/'contact/request.json').read_text())
        assert len(request['image_paths'])==len(anchors['images'])
        assert 'manual_good' not in json.dumps(request)
        subprocess.run(base+['--stages','paths','--run'],cwd=ROOT,env=env,check=True)
        result=json.loads((out/'paths/result.json').read_text())
        assert result['success'] and result['mask']['length_mm']>0
        assert result['depth_point_cloud']['valid'],result
        print('PASS: synthetic CPU mask/anchor/path integration; contact was mocked, no inference.')


if __name__=='__main__':main()
