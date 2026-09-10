#!/usr/bin/env python3
"""Four isolated inference conditions on saved Contact89; no perception or hardware."""
import argparse
import base64
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'reference/src'))
from upv_vlm_v2.prompts.anchor_ranking_prompts import get_anchor_ranking_prompt_variant

VARIANTS = ('original', 'compact', 'reduced_pixels', 'combined')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csv_save(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def request(base, endpoint, body=None):
    req = Request(base+endpoint, data=json.dumps(body).encode() if body is not None else None,
                  headers={'Content-Type':'application/json'}, method='POST' if body is not None else 'GET')
    with urlopen(req, timeout=600) as response:
        raw = response.read().decode()
        return json.loads(raw) if raw else {}


def parse(raw, ids):
    decoder = json.JSONDecoder()
    for pos, char in enumerate(raw):
        if char != '{': continue
        try: obj, _ = decoder.raw_decode(raw[pos:])
        except ValueError: continue
        if isinstance(obj, dict) and 'per_anchor_analysis' in obj: break
    else: raise ValueError('Missing per_anchor_analysis JSON')
    rows = obj['per_anchor_analysis']
    if not isinstance(rows, list) or len(rows)!=len(ids) or any(not isinstance(r,dict) for r in rows):
        raise ValueError('Incomplete candidate analysis')
    if {r.get('anchor_id') for r in rows} != set(ids): raise ValueError('Invalid/duplicate IDs')
    for r in rows:
        s=r.get('anchor_score')
        if type(r.get('anchor_usable')) is not bool or type(s) not in (int,float) or not math.isfinite(s) or not 0<=s<=100:
            raise ValueError('Invalid usability/score')
    return {r['anchor_id']:r for r in rows}


def summarize(out, predictions, timings):
    metrics=[]; selections=[]
    for variant in sorted({r['variant'] for r in predictions}):
        rows=[r for r in predictions if r['variant']==variant]
        valid=[r for r in rows if r['predicted_good'] is not None]
        tp=sum(r['manual_good'] and r['predicted_good'] for r in valid)
        fp=sum(not r['manual_good'] and r['predicted_good'] for r in valid)
        tn=sum(not r['manual_good'] and not r['predicted_good'] for r in valid)
        fn=sum(r['manual_good'] and not r['predicted_good'] for r in valid)
        div=lambda a,b:a/b if b else None
        recall,specificity=div(tp,tp+fn),div(tn,tn+fp)
        good=bad=abstain=incomplete=0
        for key in {(r['split'],r['case']) for r in rows}:
            scene=[r for r in rows if (r['split'],r['case'])==key]
            complete=all(r['predicted_good'] is not None for r in scene)
            usable=sorted([r for r in scene if r['predicted_good']],key=lambda r:(-r['score'],int(r['anchor_id'][1:])))
            chosen=usable[0] if complete and usable else None
            selections.append(dict(variant=variant,split=key[0],case=key[1],
                action='INPUT_FAILURE' if not complete else 'SELECT' if chosen else 'NO_SAFE_ANCHOR',
                selected_anchor_id=chosen['anchor_id'] if chosen else None,selected_good=chosen['manual_good'] if chosen else None))
            if not complete:incomplete+=1;continue
            if not usable: abstain+=1
            elif usable[0]['manual_good']:good+=1
            else:bad+=1
        times=[r['request_ms']/1000 for r in timings if r['variant']==variant and r['valid']]
        metrics.append(dict(variant=variant,n_total=len(rows),n_valid=len(valid),TP=tp,FP=fp,TN=tn,FN=fn,
            precision=div(tp,tp+fp),recall=recall,specificity=specificity,F1=div(2*tp,2*tp+fp+fn),
            accuracy=div(tp+tn,len(valid)),balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
            good_selections=good,bad_selections=bad,abstain=abstain,incomplete_scenes=incomplete,
            selection_usability=div(good,good+bad),mean_request_s=statistics.mean(times) if times else None,
            median_request_s=statistics.median(times) if times else None))
    csv_save(out/'predictions.csv',predictions); csv_save(out/'timings.csv',timings);csv_save(out/'metrics.csv',metrics)
    csv_save(out/'selections.csv',selections)
    save(out/'metrics.json',metrics)
    text='# Fast contact experiment\n\nSaved-crop inference only; NOT full-pipeline or robot timing. Historical reference is not a fresh control.\n\n'
    text+='| Variant | Valid/total | Precision | Recall | Specificity | F1 | Accuracy | Balanced accuracy | Good/bad selections | Mean request s |\n|---|---|---|---|---|---|---|---|---|---|\n'
    for m in metrics:
        vals=['NA' if m[k] is None else f'{m[k]*100:.2f}%' for k in ('precision','recall','specificity','F1','accuracy','balanced_accuracy')]
        text+=f"| {m['variant']} | {m['n_valid']}/{m['n_total']} | "+' | '.join(vals)+f" | {m['good_selections']}/{m['bad_selections']} | {m['mean_request_s']} |\n"
    (out/'comparison.md').write_text(text)
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true',help='Explicitly send inference requests; default validates only')
    p.add_argument('--server-url',default='http://127.0.0.1:8895')
    p.add_argument('--max-cases',type=int,default=0,help='0 means all 21 scenes / 89 pairs')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--sleep-between-cases',action='store_true',help='Exercise server sleep/wake; no perception is run during sleep')
    p.add_argument('--reduced-max-pixels',type=int,default=200704)
    a=p.parse_args()
    if a.max_cases<0 or a.reduced_max_pixels<3136:p.error('Invalid case count or pixel budget')
    base=ROOT/'datasets/fast_contact89'
    data=json.loads((base/'manifest.json').read_text())
    for r in data:
        image=base/r['image']
        if hashlib.sha256(image.read_bytes()).hexdigest()!=r['image_sha256']:raise ValueError(f'Image mismatch: {image}')
    keys=sorted({(r['split'],r['case']) for r in data})
    if a.max_cases:keys=keys[:a.max_cases]
    data=[r for r in data if (r['split'],r['case']) in keys]
    print(f'{len(data)} pairs, {len(keys)} scenes, {len(keys)*len(a.variants)} planned requests',flush=True)
    if not a.run:return
    out=ROOT/'outputs/fast_contact_experiment'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    save(out/'settings.json',vars(a));save(out/'input_manifest.json',data)
    health=request(a.server_url,'/v1/models');save(out/'server_models.json',health)
    if not any(r['id']=='contact-nf4' for r in health['data']):raise ValueError('Wrong served model name')
    predictions=[dict(variant='historical_qwen32',split=r['split'],case=r['case'],anchor_id=r['anchor_id'],manual_good=r['manual_good'],predicted_good=r['reference_good'],score=r['reference_score']) for r in data]
    timings=[];schedule=[(v,k) for k in keys for v in a.variants];random.Random(12345).shuffle(schedule)
    save(out/'schedule.json',schedule)
    # Prepopulate unattempted rows so an interrupted experiment cannot look complete.
    for v in a.variants:
        predictions.extend(dict(variant=v,split=r['split'],case=r['case'],anchor_id=r['anchor_id'],manual_good=r['manual_good'],predicted_good=None,score=None) for r in data)
    for i,(variant,key) in enumerate(schedule,1):
        rr=sorted([r for r in data if [r['split'],r['case']]==list(key)],key=lambda r:int(r['anchor_id'][1:]));ids=[r['anchor_id'] for r in rr]
        folder=out/f'{i:03d}_{variant}';folder.mkdir()
        system,user=get_anchor_ranking_prompt_variant('multi_image_v1_original',ids)
        if variant in ('compact','combined'):
            system+='\nOUTPUT OVERRIDE: Return only per_anchor_analysis, one object per image with anchor_id, anchor_usable (boolean), anchor_score (0-100). Apply the same contact criteria. Omit explanations, side fields, rankings and final selection.'
            user+='\nUse the compact schema specified in the output override, not the earlier full schema.'
        content=[]
        for r in rr:
            content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode((base/r['image']).read_bytes()).decode()}})
        content.append({'type':'text','text':user})
        payload=dict(model='contact-nf4',messages=[{'role':'system','content':system},{'role':'user','content':content}],temperature=0,seed=12345,max_tokens=1600)
        if variant in ('reduced_pixels','combined'):payload['mm_processor_kwargs']={'min_pixels':3136,'max_pixels':a.reduced_max_pixels}
        save(folder/'request.json',payload);(folder/'system_prompt.txt').write_text(system);(folder/'user_prompt.txt').write_text(user)
        timing=dict(variant=variant,split=key[0],case=key[1],request_ms=None,wake_ms=0,sleep_ms=0,valid=False,error=None,prompt_tokens=None,completion_tokens=None)
        print(f'[{i}/{len(schedule)}] {variant} {key}',flush=True)
        started=None
        try:
            if a.sleep_between_cases:
                t=time.perf_counter();request(a.server_url,'/sleep?level=1',{});timing['sleep_ms']=(time.perf_counter()-t)*1000
                t=time.perf_counter();request(a.server_url,'/wake_up',{});timing['wake_ms']=(time.perf_counter()-t)*1000
            started=time.perf_counter();response=request(a.server_url,'/v1/chat/completions',payload);timing['request_ms']=(time.perf_counter()-started)*1000
            save(folder/'response.json',response)
            text=response['choices'][0]['message']['content'];(folder/'raw_response.txt').write_text(text)
            parsed=parse(text,ids);save(folder/'parsed.json',parsed)
            usage=response.get('usage',{});timing.update(valid=True,prompt_tokens=usage.get('prompt_tokens'),completion_tokens=usage.get('completion_tokens'))
            for r in predictions:
                if r['variant']==variant and [r['split'],r['case']]==list(key):r.update(predicted_good=parsed[r['anchor_id']]['anchor_usable'],score=parsed[r['anchor_id']]['anchor_score'])
        except Exception as exc:
            timing['error']=f'{type(exc).__name__}: {exc}'
            if started is not None and timing['request_ms'] is None:timing['request_ms']=(time.perf_counter()-started)*1000
            if hasattr(exc,'read'):(folder/'http_error.txt').write_bytes(exc.read())
        timings.append(timing);save(folder/'timing.json',timing);summary=summarize(out,predictions,timings)
        if timing['error']:
            print('Stopped on error:',timing['error']);break
    print(summary);print('Output:',out)


if __name__=='__main__':main()
