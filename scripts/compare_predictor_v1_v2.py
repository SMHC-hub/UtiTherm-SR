#!/usr/bin/env python3
"""Paired V1/V2 UtiTherm-SR comparison with sequence-cluster bootstrap.

Uses independently saved Step 7A and Step 7B2C per-image detection and
per-trial timing CSVs. Results are exploratory on the reused FLIR validation
split. No model inference, re-training or checkpoint changes occur.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

METHODS = (
    'linear_640', 'full_bicubic_640', 'full_fsrcnn_640',
    'fixed_bicubic4', 'random_bicubic4', 'predictor_bicubic4',
    'random_batched_fsrcnn4', 'predictor_batched_fsrcnn4',
)
PREDICTOR_MODES = ('predictor_bicubic4', 'predictor_batched_fsrcnn4')
CONTROLS = tuple(m for m in METHODS if m not in PREDICTOR_MODES)
COUNT_FIELDS = ('small_tp', 'all_tp', 'fp', 'small_gt')


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        absent = set(required) - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f'{path} missing columns {sorted(absent)}')
        return list(reader)


def detections(path: Path) -> dict[tuple[str, str], dict]:
    rows = read_csv(path, ('image','sequence','method',*COUNT_FIELDS))
    indexed = {}
    for r in rows:
        key = (r['image'], r['method'])
        if key in indexed:
            raise ValueError(f'Duplicate detection result {key} in {path}')
        for c in COUNT_FIELDS:
            r[c] = int(r[c])
        indexed[key] = r
    if {k[1] for k in indexed} != set(METHODS):
        raise ValueError('Detection files lack expected methods or contain unknown methods.')
    counts = {m: sum(key[1] == m for key in indexed) for m in METHODS}
    if len(set(counts.values())) != 1:
        raise ValueError(f'Unequal image counts across methods: {counts}')
    return indexed


def time_means(path: Path) -> dict[tuple[str, str], float]:
    rows = read_csv(path, ('image','sequence','repeat','method','total_ms'))
    grouped = defaultdict(list)
    repeats = set()
    for r in rows:
        key = (r['image'], r['method'], int(r['repeat']))
        if key in repeats:
            raise ValueError(f'Duplicate timing trial {key} in {path}')
        repeats.add(key)
        grouped[(r['image'],r['method'])].append(float(r['total_ms']))
    if len(set(map(len, grouped.values()))) != 1:
        raise ValueError('Unequal repeats across image/method pairs.')
    return {k: sum(x)/len(x) for k,x in grouped.items()}


def percentile(sorted_values: list[float], proportion: float) -> float:
    pos=(len(sorted_values)-1)*proportion
    idx=int(pos)
    frac=pos-idx
    if idx+1 == len(sorted_values):
        return sorted_values[idx]
    return sorted_values[idx]*(1-frac)+sorted_values[idx+1]*frac


def bootstrap_cluster_ci(rows: list[dict], key: str, iterations: int, seed: int) -> list[float]:
    byseq = defaultdict(list)
    for r in rows:
        byseq[r['sequence']].append(r[key])
    seqs = sorted(byseq)
    n = len(seqs)
    sums = [sum(byseq[s]) for s in seqs]
    sizes = [len(byseq[s]) for s in seqs]
    rng = random.Random(seed)
    b = []
    for _ in range(iterations):
        chosen = [rng.randrange(n) for _ in range(n)]
        b.append(sum(sums[i] for i in chosen)/sum(sizes[i] for i in chosen))
    b.sort()
    return [round(percentile(b,.025),6), round(percentile(b,.975),6)]


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v1',type=Path,default=Path('reports/development/eval_200_v1'))
    ap.add_argument('--v2',type=Path,default=Path('reports/development/eval_200_predictor_v2_v1'))
    ap.add_argument('--out',type=Path,default=Path('reports/development/paired_v1_v2_200_v1'))
    ap.add_argument('--bootstrap',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=73106)
    ap.add_argument('--fp-penalty',type=float,default=.25)
    args=ap.parse_args()
    if args.bootstrap<1000 or args.fp_penalty < 0:
        ap.error('bootstrap must be >=1000 and fp penalty must be nonnegative')
    if args.out.exists():
        ap.error(f'Output directory exists; use a new --out to avoid overwriting: {args.out}')

    det_a=detections(args.v1/'detection_per_image.csv')
    det_b=detections(args.v2/'detection_per_image.csv')
    if set(det_a)!=set(det_b):
        raise ValueError('Runs have different (image,method) keys; comparison is invalid')
    images = sorted({im for im,_ in det_a})
    if len(images)!=200:
        raise ValueError(f'Expected frozen 200-image run, found {len(images)}')
    for key in det_a:
        if det_a[key]['sequence'] != det_b[key]['sequence']:
            raise ValueError(f'Sequence mismatch for {key}')
        if det_a[key]['small_gt'] != det_b[key]['small_gt']:
            raise ValueError(f'Ground-truth count mismatch for {key}')
    for image in images:
        first=det_a[(image,METHODS[0])]
        for method in METHODS[1:]:
            r=det_a[(image,method)]
            if r['sequence'] != first['sequence'] or r['small_gt'] != first['small_gt']:
                raise ValueError(f'Within-run sequence/GT mismatch for {image}')
    summary_a=json.loads((args.v1/'summary.json').read_text(encoding='utf-8'))
    summary_b=json.loads((args.v2/'summary.json').read_text(encoding='utf-8'))
    if summary_a.get('split_manifest_sha256') != summary_b.get('split_manifest_sha256') or not summary_a.get('split_manifest_sha256'):
        raise ValueError('Frozen split manifest hashes missing or do not match')

    seqs=sorted({det_a[(im,METHODS[0])]['sequence'] for im in images})
    if len(seqs)!=17:
        raise ValueError(f'Expected 17 development sequences, found {len(seqs)}')
    timing_a=time_means(args.v1/'timing_per_trial.csv')
    timing_b=time_means(args.v2/'timing_per_trial.csv')
    if set(timing_a)!=set(det_a) or set(timing_b)!=set(det_b):
        raise ValueError('Timing/detection keys do not match')

    output = {
        'task':'Step 7B2D paired V1 vs V2 diagnostic (exploratory)',
        'utc':datetime.now(timezone.utc).isoformat(),
        'v1_dir':str(args.v1),'v2_dir':str(args.v2),
        'input_sha256':{str(p):sha256(p) for p in (
            args.v1/'summary.json',args.v2/'summary.json',
            args.v1/'detection_per_image.csv',args.v2/'detection_per_image.csv',
            args.v1/'timing_per_trial.csv',args.v2/'timing_per_trial.csv')},
        'frozen_manifest_sha256':summary_a['split_manifest_sha256'],
        'images':len(images),'sequences':len(seqs),'seed':args.seed,
        'cluster_bootstrap_iterations':args.bootstrap,'fp_penalty':args.fp_penalty,
        'comparison':{},'control_drift':{},'timing':{},
        'limitations':[
            'This FLIR validation-domain set was used previously for research development; NOT a pristine final test.',
            'Paired cluster bootstrap resamples only 17 filename-derived video-sequence groups.',
            'Diagnostic class-aware TP/FP at fixed thresholds, NOT official mAP.',
            'V1 and V2 are separate inference runs; unchanged controls may drift, and timing is environment-sensitive.',
            'Both predictors were trained using detector-training-domain utility labels, not out-of-fold detector labels.',
            'ROI budget is four tiles (4/16), using Bicubic or FSRCNN interventions as specified.',
        ]
    }
    print(f'PAIRED AUDIT OK — {len(images)} images, {len(seqs)} sequences, matching frozen-manifest SHA256')
    gt=sum(det_a[(im,METHODS[0])]['small_gt'] for im in images)
    for method in PREDICTOR_MODES:
        paired=[]
        for image in images:
            a,b=det_a[(image,method)],det_b[(image,method)]
            dtp=b['small_tp']-a['small_tp']
            dfp=b['fp']-a['fp']
            paired.append({'image':image,'sequence':a['sequence'],
                           'd_small_tp':dtp, 'd_fp':dfp,
                           'd_utility':dtp - args.fp_penalty*dfp})
        tp1=sum(det_a[(im,method)]['small_tp'] for im in images)
        tp2=sum(det_b[(im,method)]['small_tp'] for im in images)
        fp1=sum(det_a[(im,method)]['fp'] for im in images)
        fp2=sum(det_b[(im,method)]['fp'] for im in images)
        diff_tp=tp2-tp1
        diff_fp=fp2-fp1
        diff_u=sum(r['d_utility'] for r in paired)
        record = {
            'v1':{'small_tp':tp1,'fp':fp1,'small_recall':tp1/gt},
            'v2':{'small_tp':tp2,'fp':fp2,'small_recall':tp2/gt},
            'v2_minus_v1':{
                'small_tp':diff_tp,'false_positives':diff_fp,
                'utility_total':round(diff_u,4),'utility_per_image':round(diff_u/len(images),6),
                '95ci_utility_per_image':bootstrap_cluster_ci(paired,'d_utility',args.bootstrap,args.seed),
                '95ci_small_tp_per_image':bootstrap_cluster_ci(paired,'d_small_tp',args.bootstrap,args.seed+1),
                '95ci_fp_per_image':bootstrap_cluster_ci(paired,'d_fp',args.bootstrap,args.seed+2),
                'sequence_utility_win_tie_loss':{
                    'wins':sum(sum(r['d_utility'] for r in paired if r['sequence']==s)>0 for s in seqs),
                    'ties':sum(sum(r['d_utility'] for r in paired if r['sequence']==s)==0 for s in seqs),
                    'losses':sum(sum(r['d_utility'] for r in paired if r['sequence']==s)<0 for s in seqs),
                }
            }
        }
        output['comparison'][method]=record
        d=record['v2_minus_v1']
        print(f'\n{method}:')
        print(f'  V1 small TP={tp1}, FP={fp1}; V2 small TP={tp2}, FP={fp2}')
        print(f'  V2–V1 ΔTP={diff_tp:+d}, ΔFP={diff_fp:+d}, '
              f'Δutility={diff_u:+.2f} ({diff_u/len(images):+.4f}/image)')
        print(f"  Sequence-bootstrap 95% CI, Δutility/image: {d['95ci_utility_per_image']}")
        print(f"  Sequence utility wins/ties/losses: {d['sequence_utility_win_tie_loss']}")

    print('\nUNCHANGED CONTROLS (cross-run drift audit):')
    control_changes=0
    for method in CONTROLS:
        changed=sum(any(det_a[(im,method)][c]!=det_b[(im,method)][c]
                        for c in ('small_tp','all_tp','fp')) for im in images)
        dtp=sum(det_b[(im,method)]['small_tp']-det_a[(im,method)]['small_tp'] for im in images)
        dfp=sum(det_b[(im,method)]['fp']-det_a[(im,method)]['fp'] for im in images)
        output['control_drift'][method]={'changed_images':changed,'delta_small_tp':dtp,'delta_fp':dfp}
        if changed:
            control_changes+=1
        print(f'  {method:28} changed images={changed:3} ΔTP={dtp:+d} ΔFP={dfp:+d}')

    print('\nLATENCY (separate runs; descriptive only, ms/image):')
    for method in ('linear_640','full_bicubic_640',*PREDICTOR_MODES):
        ms1=sum(timing_a[(im,method)] for im in images)/len(images)
        ms2=sum(timing_b[(im,method)] for im in images)/len(images)
        output['timing'][method]={'v1_ms':round(ms1,5),'v2_ms':round(ms2,5),'between_run_delta_ms':round(ms2-ms1,5)}
        print(f'  {method:28} V1={ms1:.3f} V2={ms2:.3f} (between-run Δ={ms2-ms1:+.3f})')
    for method in PREDICTOR_MODES:
        output['timing'][method]['v1_excess_over_full_bicubic_ms'] = round(
            output['timing'][method]['v1_ms']-output['timing']['full_bicubic_640']['v1_ms'],5)
        output['timing'][method]['v2_excess_over_full_bicubic_ms'] = round(
            output['timing'][method]['v2_ms']-output['timing']['full_bicubic_640']['v2_ms'],5)
    if control_changes:
        print(f'\nCAUTION: {control_changes}/{len(CONTROLS)} unchanged controls changed between runs; '
              'interpret tiny predictor improvements in context of run-to-run drift.')
    else:
        print('\nUNCHANGED CONTROLS: per-image detections matched between runs.')
    args.out.mkdir(parents=True,exist_ok=False)
    target=args.out/'summary.json'
    target.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(f'\nSAVED — {target}')


if __name__ == '__main__':
    main()
