#!/usr/bin/env python3
"""CPU-only contact-crop RGB/mask baseline. Defaults to development pilot."""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/contact_anchor_selection_benchmark_E3_E45_82'
PILOT = {'case_001_brick_01_clean_regular', 'case_002_brick_02_chipped_jagged',
         'case_009_timber_04_debris_obstruction', 'session_20260528_035248'}


def readcsv(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def savecsv(p, rows):
    if rows:
        with p.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def geometry(row):
    meta = json.loads(resolve(row['metadata_json_path']).read_text())
    if meta.get('support_case_dir'):
        folder = resolve(meta['support_case_dir'])
    elif meta.get('source_session'):
        folder = resolve(meta['source_session']) / 'artifacts/04_anchor_selection'
    else:
        raise ValueError('Missing explicit E3/E45 provenance')
    debug = json.loads((folder / 'anchor_crop_geometry_debug.json').read_text())
    g = debug['anchors'][row['anchor_id']]
    rgbpath = resolve(g['rotated_image_path'])
    maskpath = resolve(g['rotated_mask_path']) if g.get('rotated_mask_path') else rgbpath.with_name('rotated_object_mask.png')
    if not maskpath.is_file():
        raise ValueError(f'Missing rotated GSAM2 mask: {maskpath}')
    return g, rgbpath, maskpath


def extract(row, layout):
    g, rgbpath, maskpath = geometry(row)
    source = Image.open(rgbpath).convert('RGB')
    mask = Image.open(maskpath).convert('L')
    image = Image.open(layout).convert('RGB')
    if mask.size != source.size or list(image.size) != g['image_size_px']:
        raise ValueError('RGB/mask/rendered dimensions disagree')
    panels = []
    for side in ('top', 'bottom'):
        box = list(map(int, g[f'{side}_crop_box_in_rendered_image']))
        bounds = list(map(int, g[f'{side}_local_crop_bounds']))
        if not (0 <= bounds[0] < bounds[2] <= source.width and 0 <= bounds[1] < bounds[3] <= source.height):
            raise ValueError('Crop outside saved rotated frame')
        size = (box[2]-box[0], box[3]-box[1])
        m = mask.crop(bounds).resize(size, Image.Resampling.NEAREST)
        reference = source.crop(bounds).resize(size, Image.Resampling.BICUBIC)
        if side == 'bottom' and g.get('bottom_crop_rotated_180'):
            m = m.transpose(Image.Transpose.ROTATE_180)
            reference = reference.transpose(Image.Transpose.ROTATE_180)
        rgb = np.asarray(image.crop(box)).copy()
        guide = (rgb[:,:,0] > 180) & (rgb[:,:,2] > 180) & (rgb[:,:,1] < 100)
        counts = guide.sum(axis=1)
        if counts.max() < .5 * size[0]:
            raise ValueError('Expected per-panel magenta guide unavailable')
        y = int(np.median(np.flatnonzero(counts > .5 * size[0])))
        exclude = cv2.dilate(guide.astype('uint8'), np.ones((9,9), 'uint8')) > 0
        # Photometric check catches incompatible metadata without using labels.
        difference = np.abs(rgb.astype(float)-np.asarray(reference).astype(float)).mean(axis=2)
        mae = float(difference[~exclude].mean())
        if mae > 15:
            raise ValueError(f'Crop provenance mismatch: reconstruction MAE={mae:.2f}')
        panels.append((side, rgb, np.asarray(m)>127, exclude, y, mae))
    return panels, maskpath


def analyse(rgb, mask, exclude, y, args):
    h,w = mask.shape
    yy,xx = np.indices(mask.shape)
    transition = mask[1:] != mask[:-1]
    profile = np.full(w, np.nan)
    for x in range(w):
        hits = np.flatnonzero(transition[:,x])
        # Search the full crop height; choose the boundary nearest the guide.
        if len(hits):
            profile[x] = hits[np.argmin(np.abs(hits-y))] + .5
    good = np.isfinite(profile)
    if good.sum() < max(5, w//3):
        raise ValueError('Insufficient mask boundary support')
    # Deterministic robust fit; no ground-truth labels enter this function.
    fit = cv2.fitLine(np.column_stack((np.arange(w)[good], profile[good])).astype('float32'),
                      cv2.DIST_HUBER, 0, .01, .01).ravel()
    if abs(fit[0]) < .1:
        raise ValueError('Unexpected near-vertical contact rim')
    expected = fit[3] + (np.arange(w)-fit[2])*fit[1]/fit[0]
    tolerance = max(2., args.defect_fraction*w)
    residual = np.abs(profile-expected)
    defects = (~good) | (residual > tolerance)
    runs = np.diff(np.r_[False, defects, False].astype(int))
    longest = max(np.flatnonzero(runs==-1)-np.flatnonzero(runs==1), default=0)
    smooth = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5,5), 1)
    smooth = cv2.createCLAHE(2., (8,8)).apply(smooth)
    edges = cv2.Canny(smooth,60,160)>0
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1,0)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0,1)
    horizontal = np.abs(gy) >= 2*np.abs(gx)
    rim = np.abs(yy-expected[None,:]) <= tolerance
    band = np.abs(yy-y) <= max(6, round(.12*w))
    border = (xx>=5)&(xx<w-5)&(yy>=5)&(yy<h-5)
    retained = edges & band & border & ~exclude & ~(rim & horizontal)
    n, components, stats, _ = cv2.connectedComponentsWithStats(retained.astype('uint8'),8)
    obstruction = np.zeros_like(mask)
    for k in range(1,n):
        pts = components==k
        intersects = bool(np.any(pts & rim))
        if intersects and stats[k,cv2.CC_STAT_AREA]>=8 and stats[k,cv2.CC_STAT_HEIGHT]>2*tolerance:
            obstruction |= pts
    mask_bad = longest/w > args.max_defect_run_fraction
    edge_bad = bool(obstruction.any())
    score = float(longest/w + obstruction.sum()/max(1,band.sum()))
    overlay = rgb.copy()
    overlay[mask] = (.7*overlay[mask]+.3*np.array([0,180,90])).astype('uint8')
    for x in range(w):
        if good[x]:
            cv2.circle(overlay,(x,int(profile[x])),1,(255,60,0) if defects[x] else (0,255,255),-1)
    evidence = rgb.copy()
    evidence[exclude] = (110,110,110)
    evidence[obstruction] = (255,0,0)
    return {'good':not(mask_bad or edge_bad), 'score':score,
            'mask_defect_run_fraction':float(longest/w), 'rgb_obstruction_pixels':int(obstruction.sum()),
            'mask_bad':bool(mask_bad),'edge_bad':edge_bad}, [rgb, overlay, np.repeat((edges*255).astype('uint8')[:,:,None],3,axis=2), evidence]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['pilot','full'], default='pilot')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--defect-fraction',type=float,default=.03)
    p.add_argument('--max-defect-run-fraction',type=float,default=.10)
    a=p.parse_args()
    rows=readcsv(DATA/'dataset_manifest.csv')
    layouts={r['benchmark_id']:resolve(r['layout_image_path']) for r in readcsv(DATA/'layouts/L6_magenta_line_only_fixed/layout_manifest.csv')}
    if a.mode=='pilot':
        rows=[r for r in rows if r['case_or_session_id'] in PILOT]
    out=a.output_dir or ROOT/'outputs/contact_crop_rgb_mask_baseline'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    config={**vars(a),'output_dir':str(out),'development_scenes':sorted(PILOT),
            'status':'development; thresholds unvalidated','depth_used':False,'learned_contact_classifier':False}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    predictions=[]; failures=[]; sides=[]
    for row in rows:
        bid=row['benchmark_id']
        try:
            panels,maskpath=extract(row,layouts[bid])
            if a.dry_run:
                print(bid, 'alignment OK',flush=True)
                continue
            results=[]; tiles=[]
            for side,rgb,mask,exclude,y,mae in panels:
                result, images=analyse(rgb,mask,exclude,y,a)
                results.append(result)
                sides.append({'benchmark_id':bid,'side':side,'alignment_mae':mae,**result})
                for title,im in zip(['RGB crop','Mask boundary / defects','Canny','Guide excluded / obstruction'],images):
                    tile=Image.new('RGB',(max(320,im.shape[1]),im.shape[0]+50),'white')
                    tile.paste(Image.fromarray(im),(0,50));ImageDraw.Draw(tile).text((8,8),side+' | '+title,fill='black')
                    tiles.append(tile)
            canvas=Image.new('RGB',(sum(t.width for t in tiles[:4]),max(t.height for t in tiles)*2),'white')
            for i,t in enumerate(tiles):
                canvas.paste(t,(sum(q.width for q in tiles[(i//4)*4:i]),(i//4)*max(q.height for q in tiles)))
            (out/'visuals').mkdir(exist_ok=True);canvas.save(out/'visuals'/f'{bid}.png')
            predictions.append({'benchmark_id':bid,'scene':row['case_or_session_id'],'anchor_id':row['anchor_id'],
                                'development':row['case_or_session_id'] in PILOT,
                                'manual_good':row['is_usable']=='True','predicted_good':all(r['good'] for r in results),
                                'risk_score':max(r['score'] for r in results),'layout_path':str(layouts[bid]),
                                'layout_sha256':hashlib.sha256(layouts[bid].read_bytes()).hexdigest(),'mask_path':str(maskpath)})
        except (ValueError,KeyError,OSError) as e:
            failures.append({'benchmark_id':bid,'reason':str(e)})
    savecsv(out/'predictions.csv',predictions);savecsv(out/'side_diagnostics.csv',sides);savecsv(out/'failures.csv',failures)
    selected=[]
    for scene in sorted({r['scene'] for r in predictions}):
        group=[r for r in predictions if r['scene']==scene]
        passing=sorted([r for r in group if r['predicted_good']],key=lambda r:(r['risk_score'],r['anchor_id']))
        incomplete=any(r['case_or_session_id']==scene and r['benchmark_id'] in {f['benchmark_id'] for f in failures} for r in rows)
        chosen=passing[0] if passing and not incomplete else None
        selected.append({'scene':scene,'selected_anchor':chosen['anchor_id'] if chosen else '',
                         'action':'INPUT_FAILURE' if incomplete else ('SELECT' if chosen else 'NO_SAFE_ANCHOR'),
                         'selected_good':chosen['manual_good'] if chosen else ''})
    savecsv(out/'selections.csv',selected)
    metrics=[]
    for name, subset in [('all',predictions),('development',[r for r in predictions if r['development']]),('held_out',[r for r in predictions if not r['development']])]:
        tp=sum(r['manual_good'] and r['predicted_good'] for r in subset);fp=sum(not r['manual_good'] and r['predicted_good'] for r in subset)
        tn=sum(not r['manual_good'] and not r['predicted_good'] for r in subset);fn=sum(r['manual_good'] and not r['predicted_good'] for r in subset)
        div=lambda x,y:x/y if y else None
        recall=div(tp,tp+fn);specificity=div(tn,tn+fp)
        metrics.append(dict(group=name,n=len(subset),TP=tp,FP=fp,TN=tn,FN=fn,precision=div(tp,tp+fp),recall=recall,specificity=specificity,F1=div(2*tp,2*tp+fp+fn),accuracy=div(tp+tn,len(subset)),balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None))
    savecsv(out/'metrics.csv',metrics)
    (out/'README.md').write_text('# RGB and shared GSAM2 mask contact baseline\n\nDevelopment implementation; do not present as SOTA or validated paper results.\n\nUses exact saved L6 RGB panel boxes and transformed saved rotated GSAM2 masks. Photometric reconstruction check rejects mismatched provenance. The mask contour supplies robust expected-rim and coherent notch/protrusion evidence. RGB Canny components provide additional obstruction evidence. Magenta neighbourhood is excluded; hidden pixels are not reconstructed. Thresholds are crop-relative, not millimetres.\n\nPair passes only if both sides pass. Missing inputs are failures. Pilot scenes are development scenes; freeze parameters before full evaluation. The remaining subset has also been inspected in earlier project work and is not a pristine prospective test set.\n\nVisual columns: RGB, aligned mask contour (cyan normal/orange defect), Canny, excluded guide (grey) and obstruction evidence (red).\n')
    print(f'Processed={len(predictions)} failures={len(failures)} output={out}',flush=True)


if __name__=='__main__':
    main()
