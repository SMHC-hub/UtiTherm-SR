"""Step 7A: evaluation of a FROZEN FLIR validation-domain development split.

Paired controls: linear, full bicubic, full FSRCNN, fixed/random/predictor
ROI Bicubic, random/predictor batched ROI FSRCNN. All models fixed.

This is a development evaluation, NOT a pristine final test set: Step 4 has
already examined this FLIR validation domain. Diagnostic recall IoU=.5,
confidence=.25; no official mAP computed in this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from prepare_development_split import sequence_id, training_sequences, check_manifest, sha256
from evaluate_selective_sr import predictor_model, predict_tile_scores, choose_tiles
from probe_roi_utility import roi_grid, prediction_records, detection_counts
from evaluate_bicubic_baseline import ground_truth
from benchmark_selective_fsrcnn import load_sr_checkpoint
from utitherm.sr.roi_bicubic import render_selected
from utitherm.sr.roi_fsrcnn_batched import render_batched_fsrcnn
from utitherm.sr.roi_fsrcnn import render_fsrcnn

MODES = (
    'linear_640', 'full_bicubic_640', 'full_fsrcnn_640',
    'fixed_bicubic4', 'random_bicubic4', 'predictor_bicubic4',
    'random_batched_fsrcnn4', 'predictor_batched_fsrcnn4',
)
HR = (640, 512)


def assigned_tiles(method: str, name: str, scores: np.ndarray, seed: int) -> list[int]:
    if method.startswith('fixed_'):
        return choose_tiles('fixed_first4', name, 4, 16, seed)
    if method.startswith('random_'):
        return choose_tiles('random4', name, 4, 16, seed)
    if method.startswith('predictor_'):
        return choose_tiles('predictor4', name, 4, 16, seed, scores)
    return []


def render(method: str, gray: np.ndarray, tiles: list, selected: list[int], sr,
           device: torch.device) -> tuple[np.ndarray, int]:
    if method == 'linear_640':
        return cv2.resize(gray, HR, interpolation=cv2.INTER_LINEAR), 0
    if method == 'full_bicubic_640':
        return cv2.resize(gray, HR, interpolation=cv2.INTER_CUBIC), 640*512
    if method == 'full_fsrcnn_640':
        return render_fsrcnn(gray, tiles, [], sr, device, 'full')
    if method.endswith('_bicubic4'):
        return render_selected(gray, tiles, selected, feather=8, roi_only=True)
    if method.endswith('_batched_fsrcnn4'):
        return render_batched_fsrcnn(gray, tiles, selected, sr, device, feather=8, halo=16)
    raise ValueError(f'Unknown method: {method}')


def time_trial(method, gray, name, tiles, predictor, input_size, sr,
               detector, cfg, device, expected) -> tuple[dict, list[dict]]:
    if method not in MODES:
        raise ValueError('Unexpected evaluation mode')
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter_ns()
    scores = None
    if method.startswith('predictor_'):
        scores = predict_tile_scores(predictor, gray, tiles, input_size,
                                      device, float(cfg['false_positive_penalty']))
    chosen = assigned_tiles(method, name, scores, int(cfg['seed']))
    if tuple(chosen) != expected[method]:
        raise RuntimeError('Selection changed during a repeated benchmark trial')
    torch.cuda.synchronize(device)
    t1 = time.perf_counter_ns()
    output, computed = render(method, gray, tiles, chosen, sr, device)
    torch.cuda.synchronize(device)
    t2 = time.perf_counter_ns()
    preds = prediction_records(detector, output, cfg)
    torch.cuda.synchronize(device)
    t3 = time.perf_counter_ns()
    return {'selector_ms':(t1-t0)/1e6, 'sr_ms':(t2-t1)/1e6,
            'detector_ms':(t3-t2)/1e6, 'total_ms':(t3-t0)/1e6,
            'computed_pixels':computed,
            'peak_torch_mib':torch.cuda.max_memory_allocated(device)/1024**2}, preds


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('Refuse empty CSV')
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def analysis(trials: list[dict], det_rows: list[dict], seed: int, n_boot: int = 1000) -> dict:
    """Bootstrap SEQUENCE clusters, not repeated timings or correlated frames."""
    if not trials or not det_rows or n_boot < 100:
        raise ValueError('Incomplete evaluation input')
    by_trial = {}
    for row in trials:
        key = row['image'], row['method'], int(row['repeat'])
        if key in by_trial:
            raise ValueError('Duplicate trial')
        by_trial[key] = row
    images = sorted({r['image'] for r in det_rows})
    reps = sorted({int(r['repeat']) for r in trials})
    expected = {(i,m,r) for i in images for m in MODES for r in reps}
    if set(by_trial) != expected:
        raise ValueError('Missing trials')
    per_image = defaultdict(dict)
    for row in det_rows:
        key = row['image']; method = row['method']
        if method in per_image[key]:
            raise ValueError('Duplicate detection measurements')
        per_image[key][method] = row
    if any(set(d) != set(MODES) for d in per_image.values()):
        raise ValueError('Missing policies for detection')
    by_seq = defaultdict(list)
    for im in images:
        by_seq[sequence_id(im)].append(im)
    if len(by_seq) < 2:
        raise ValueError('Need >=2 independent sequence clusters')
    rng = np.random.default_rng(seed)
    result = {}
    timed = {}
    for mode in MODES:
        subset = [per_image[im][mode] for im in images]
        cnt = {k:sum(int(row[k]) for row in subset) for k in ('small_gt','small_tp','all_tp','fp')}
        per_img_median = {im:statistics.median(float(by_trial[(im,mode,r)]['total_ms']) for r in reps)
                          for im in images}
        timed[mode] = per_img_median
        result[mode] = {**cnt, 'small_recall_conf_025':round(cnt['small_tp']/cnt['small_gt'],6) if cnt['small_gt'] else None,
                        'mean_of_image_median_total_ms':round(statistics.mean(per_img_median.values()),4),
                        'mean_selector_ms':round(statistics.mean(statistics.median(
                            float(by_trial[(im,mode,r)]['selector_ms']) for r in reps) for im in images),4),
                        'mean_sr_ms':round(statistics.mean(statistics.median(
                            float(by_trial[(im,mode,r)]['sr_ms']) for r in reps) for im in images),4)}
    base = result['linear_640']
    seq_names = sorted(by_seq)
    comparisons = {}
    for mode in MODES[1:]:
        delta = (result[mode]['small_tp'] - base['small_tp'] -
                 0.25*(result[mode]['fp'] - base['fp']))
        # Build one value per IMAGE for sequence-aware resampling.
        time_by_seq=[]; utility_by_seq=[]
        for seq in seq_names:
            ims = by_seq[seq]
            time_by_seq.append(statistics.mean(timed[mode][i] - timed['full_bicubic_640'][i] for i in ims))
            utility_by_seq.append(statistics.mean(
                int(per_image[i][mode]['small_tp'])-int(per_image[i]['linear_640']['small_tp'])-
                .25*(int(per_image[i][mode]['fp'])-int(per_image[i]['linear_640']['fp'])) for i in ims))
        ci_time=[]; ci_utility=[]
        for series in (time_by_seq, utility_by_seq):
            arr=np.asarray(series)
            boot=[rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
            val=[round(float(np.percentile(boot,2.5)),4),round(float(np.percentile(boot,97.5)),4)]
            if series is time_by_seq: ci_time=val
            else: ci_utility=val
        comparisons[mode] = {
            'utility_total_vs_linear':round(delta,4),
            'utility_per_image_vs_linear':round(delta/len(images),5),
            'utility_per_sequence_average_bootstrap_95ci':ci_utility,
            'mean_time_delta_vs_full_bicubic_ms':round(statistics.mean(timed[mode][i]-timed['full_bicubic_640'][i] for i in images),4),
            'mean_time_delta_per_sequence_bootstrap_95ci_ms':ci_time,
        }
    return {'images':len(images), 'unique_sequences':len(by_seq), 'repeats_per_image':len(reps),
            'methods':result, 'paired_comparisons':comparisons,
            'notes':['IoU=.5, confidence=.25: diagnostic TP/FP; NO official AP in this run.',
                     'CIs resample video sequence groups, not repeat measurements.',
                     'Validation set previously studied in Step 4: development only.',
                     'FP counts may change despite max 1-level pixel differences.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/development_eval.yaml'))
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--limit',type=int,default=8,help='0=all frozen images, >0 first N for smoke')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.limit<0 or not 3<=args.repeats<=10 or args.out.exists():
        p.error('Require limit>=0, repeats=3..10, new --out')
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if str(cfg['device'])!='0' or int(cfg['imgsz'])!=640 or int(cfg['budget_tiles'])!=4 or int(cfg['feather_hr_px'])!=8:
        p.error('Step 7A requires fixed CUDA GPU=0, input 640, budget=4, feather=8')
    if any(abs(float(cfg[k])-v)>1e-8 for k,v in (('confidence',.25),('nms_iou',.7),('false_positive_penalty',.25))):
        p.error('Fixed diagnostic confidence=.25, NMS IoU=.7, false-positive penalty=.25')
    train_dir=Path(cfg['train_lr']); val_dir=Path(cfg['eval_lr']); labels=val_dir.parent.parent/'labels'/'val'
    training_seq=training_sequences(train_dir,Path(cfg['utility_csv']),Path(cfg['split_manifest']))
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    names=check_manifest(manifest,val_dir,labels,training_seq,Path(cfg['utility_csv']),Path(cfg['split_manifest']))
    for k in ('predictor_checkpoint', 'detector_checkpoint', 'sr_checkpoint'):
        if manifest['checkpoint_sha256'][k] != sha256(Path(cfg[k])):
            raise ValueError(f'Checkpoint changed since split was frozen: {k}')
    # Crucial: never infer an alternative split from a --limit; frozen manifest determines it.
    if args.limit:
        names=names[:args.limit]
    if len(names)<2:
        p.error('Need at least two frames from different sequences')
    if not torch.cuda.is_available():
        p.error('CUDA device not available')
    device=torch.device('cuda:0')
    cv2.setNumThreads(1)
    predictor,input_size=predictor_model(Path(cfg['predictor_checkpoint']),Path(cfg['utility_csv']),device)
    sr=load_sr_checkpoint(Path(cfg['sr_checkpoint']),device)
    from ultralytics import YOLO
    detector=YOLO(str(cfg['detector_checkpoint']))
    tiles=roi_grid(640,512,4,4)
    dummy=np.zeros((256,320),dtype=np.uint8)
    for _ in range(2):
        predict_tile_scores(predictor,dummy,tiles,input_size,device,.25)
        for m in MODES:
            selected=[0,5,10,15] if m.endswith('4') else []
            frame,_=render(m,dummy,tiles,selected,sr,device)
            prediction_records(detector,frame,cfg)
    torch.cuda.synchronize(device)
    trials=[]; dets=[]
    for idx,name in enumerate(names,1):
        gray=cv2.imread(str(val_dir/name),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=(256,320):
            raise ValueError(f'Invalid LR validation image: {name}')
        scores=predict_tile_scores(predictor,gray,tiles,input_size,device,.25)
        expect={m:tuple(assigned_tiles(m,name,scores,int(cfg['seed']))) for m in MODES}
        first_pass={}
        for rep in range(args.repeats):
            order=list(MODES)
            random.Random(int(cfg['seed'])+idx*1073+rep).shuffle(order)
            for m in order:
                row,pred=time_trial(m,gray,name,tiles,predictor,input_size,sr,detector,
                                     cfg,device,expect)
                trials.append({'image':name,'sequence':sequence_id(name),'repeat':rep,'method':m,
                               **{k:round(float(v),6) for k,v in row.items()}})
                if rep==0:
                    first_pass[m]=pred
        # No labels used until ALL policies have selected regions and inferred.
        gts=ground_truth(labels/(Path(name).stem+'.txt'))
        for m in MODES:
            dets.append({'image':name,'sequence':sequence_id(name),'method':m,
                         **detection_counts(gts,first_pass[m],.25)})
        if idx==1 or idx%10==0 or idx==len(names):
            print(f'Evaluated {idx}/{len(names)} frozen FLIR validation frames',flush=True)
    stats=analysis(trials,dets,int(cfg['seed']))
    checkpoints={k:{'path':str(cfg[k]),'sha256':sha256(Path(cfg[k]))}
                 for k in ('predictor_checkpoint','detector_checkpoint','sr_checkpoint')}
    final={'task':'Step 7A larger sequence-audited development evaluation',
           'created_utc':datetime.now(timezone.utc).isoformat(),
           'split_manifest_sha256':sha256(args.manifest),
           'protocol':manifest['protocol'], 'checkpoints':checkpoints,'metrics':stats,
           'limitations':['This is not a never-used final test set; FLIR validation influenced Step 4 development.',
                          'Sequence IDs derived from filenames, not independent physical scene metadata.',
                          'Diagnostic TP/FP only. Standard mAP, multiple seeds and external tests required.',
                          'Predictor trained for bicubic utility, not FSRCNN utility.',
                          'Latency excludes data IO, model loading; synchronized single-frame GPU trials.']}
    args.out.mkdir(parents=True)
    write_csv(args.out/'timing_per_trial.csv',trials)
    write_csv(args.out/'detection_per_image.csv',dets)
    (args.out/'summary.json').write_text(json.dumps(final,indent=2)+'\n',encoding='utf-8')
    print('\nSTEP 7A DEVELOPMENT RESULTS (not final held-out test):')
    for mode in MODES:
        v=stats['methods'][mode]
        print(f'  {mode:30s} small recall={v["small_recall_conf_025"]:.3f} '
              f'FP={v["fp"]:5d} time={v["mean_of_image_median_total_ms"]:.3f} ms')
    print(f'DONE — {args.out/"summary.json"}',flush=True)


if __name__=='__main__':
    main()
