"""Training-free CLIP material prototypes; batched, isolated scoring process."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

MATERIALS = ('brick', 'concrete block', 'timber')
MODE = 'mean_embedding_class_softmax'
RULE = 'clip_prompt_ensemble_requested_score_threshold'


def prompts():
    return json.loads(Path(__file__).with_name('clip_ensemble_prompts.json').read_text())


def prototype_probabilities(image_features, text_features, groups, logit_scale):
    """Normalize descriptions, average within class, normalize prototypes, softmax."""
    import torch
    text = text_features / text_features.norm(dim=-1, keepdim=True)
    means = torch.stack([text[indices].mean(0) for indices in groups])
    means = means / means.norm(dim=-1, keepdim=True)
    image = image_features / image_features.norm(dim=-1, keepdim=True)
    return (logit_scale * image @ means.T).softmax(dim=-1)


def choose(scored, requested, threshold):
    if requested not in MATERIALS or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid material or CLIP threshold')
    for row in scored:
        s = row['clip_scores']
        if set(s)!=set(MATERIALS) or any(not math.isfinite(float(v)) or not 0<=float(v)<=1 for v in s.values()):
            raise ValueError('Invalid CLIP class probabilities')
    def key(row):
        s = row['clip_scores']
        return s[requested], s[requested]-max(s[m] for m in MATERIALS if m!=requested)
    winner = max(scored, key=key) if scored else None
    selected = winner if winner and winner['clip_scores'][requested]>=threshold else None
    return winner, selected


def score_images(paths, model_id, device):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    descriptions = prompts()
    flat = [p for m in MATERIALS for p in descriptions[m]]
    groups = []
    start = 0
    for m in MATERIALS:
        groups.append(list(range(start,start+len(descriptions[m]))))
        start += len(descriptions[m])
    with torch.inference_mode():
        text = model.get_text_features(**processor(text=flat,padding=True,return_tensors='pt').to(device))
        if not isinstance(text,torch.Tensor):
            text=text.pooler_output
        output = []
        for path in paths:
            with Image.open(path) as im:
                image=im.convert('RGB')
            f=model.get_image_features(**processor(images=image,return_tensors='pt').to(device))
            if not isinstance(f,torch.Tensor):
                f=f.pooler_output
            prob=prototype_probabilities(f,text,groups,model.logit_scale.exp())[0].cpu().tolist()
            output.append(dict(zip(MATERIALS,prob)))
    return output


def verify(*, candidates, requested_material, config, output_path, crop_for_candidate, python_info):
    cfg=config.get('target_selection',{})
    threshold=float(cfg.get('clip_score_threshold',.35))
    model_id=config.get('perception',{}).get('clip_model_id','openai/clip-vit-base-patch32')
    device=config.get('perception',{}).get('clip_device','cpu')
    if not cfg.get('use_clip_crop_verifier',True):
        raise ValueError('Ensemble selection requires CLIP; detection-score fallback is disabled')
    crops=[crop_for_candidate(c) for c in candidates]
    errors=[]
    scored=[]
    try:
        if any(not path for path,_ in crops):
            raise ValueError('Candidate missing a verification crop')
        if candidates:
            request=dict(paths=[path for path,_ in crops],model_id=model_id,device=device)
            command=[python_info['clip_subprocess_python'],str(Path(__file__).resolve())]
            response=subprocess.run(command,input=json.dumps(request),capture_output=True,text=True,
                timeout=float(cfg.get('clip_timeout_sec',300)),check=False)
            if response.returncode:
                raise RuntimeError(response.stderr.strip() or 'CLIP ensemble worker failed')
            values=json.loads(response.stdout)['scores']
            if len(values)!=len(candidates):
                raise ValueError('Incomplete CLIP ensemble response')
        else:
            values=[]
        for c,(crop,kind),s in zip(candidates,crops,values):
            top=max(s,key=s.get)
            scored.append({**c,'clip_scores':s,'clip_aggregate_scores':s,'material_scores':s,
                'clip_labels':list(MATERIALS),'clip_top_label':top,'clip_top_score':s[top],
                'requested_material_score':s[requested_material],'clip_score_for_requested':s[requested_material],
                'requested_score':s[requested_material],'final_score':s[requested_material],
                'score_margin':s[requested_material]-max(s[m] for m in MATERIALS if m!=requested_material),
                'crop_used_for_verification':crop,'crop_path_used_for_scoring':crop,
                'crop_type_used_for_scoring':kind,'crop_type_used_for_verification':kind,
                'clip_score_source':'clip_crop_verifier','score_source':'clip_crop_verifier',
                'clip_error':None,'selection_rule':RULE,'class_score_aggregation':MODE})
        winner,selected=choose(scored,requested_material,threshold)
    except Exception as exc:
        errors.append(str(exc))
        winner=selected=None
    selected_id=selected['candidate_id'] if selected else 'NO_VERIFIED_MATCH'
    reason='CLIP inference failure' if errors else ('requested score >= threshold' if selected else 'no candidate or requested score below threshold')
    for row in scored:
        row['final_selected']=row['candidate_id']==selected_id
        row['final_selection_reason']=reason
    payload=dict(requested_material=requested_material,decision_backend='clip_crop_verifier',
        official_selector='clip_crop_verifier',qwen_vlm_server_enabled=False,require_vlm_decision=False,
        clip_verifier_enabled=True,clip_model_id=model_id,clip_device=device,**python_info,
        class_score_aggregation=MODE,selection_rule=RULE,threshold=threshold,prompts=prompts(),
        labels=[p for ps in prompts().values() for p in ps],candidates=scored,candidate_scores=scored,
        errors=errors,clip_verifier_failed=bool(errors),failure_reason='CLIP_VERIFIER_FAILED' if errors else None,
        selected_candidate_id=selected_id,final_selected_candidate_id=selected_id,
        no_verified_match=not selected and not errors,final_selection_reason=reason,
        allow_no_verified_match=True,absent_class_rejection_enabled=True,
        target_absent_rejection_triggered=not selected and not errors,
        target_absent_rejection_reason=reason if not selected and not errors else None,
        selected_candidate_id_before_no_match_gate=winner['candidate_id'] if winner else None,
        selected_candidate_id_after_no_match_gate=selected_id,
        source_class_prior_used=False,source_class_prior_bonus=0.,same_source_preferred=False,
        color_texture_guards_enabled=False,margin_rejection_enabled=False,
        beige_brick_guard_triggered=False,concrete_color_texture_guard_triggered=False,timber_guard_triggered=False)
    out=Path(output_path)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    return payload


if __name__=='__main__':
    request=json.load(sys.stdin)
    print(json.dumps({'scores':score_images(**request)},allow_nan=False))
