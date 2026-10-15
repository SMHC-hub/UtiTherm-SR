"""Step 6B: true ROI-only Bicubic & matched detection/latency pilot.

Run under Docker from project root:
  python scripts/benchmark_roi_bicubic.py --limit 4 \
    --out reports/utility/roi_compute_smoke_v1

The detector and predictor checkpoints are FIXED. No new model training.
Same Step 5B validation sequences as Step 6A; YOLO trained on these frames.
This is NOT an independent test set or a demonstration of learned-SR savings.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utitherm.sr.roi_bicubic import render_selected
from evaluate_selective_sr import (REQUIRED, LR_SHAPE, HR_SIZE,
                                   get_eval_image_names, predictor_model,
                                   predict_tile_scores, choose_tiles)
from probe_roi_utility import (roi_grid, prediction_records, detection_counts)
from evaluate_bicubic_baseline import ground_truth

MODES = ('linear_640', 'full_bicubic_640', 'fullframe_selective4', 'roi_only_selective4')


def make_frame(gray, tiles, chosen, mode, feather, halo):
    if mode == 'linear_640':
        return cv2.resize(gray, HR_SIZE, interpolation=cv2.INTER_LINEAR), 0
    if mode == 'full_bicubic_640':
        frame = cv2.resize(gray, HR_SIZE, interpolation=cv2.INTER_CUBIC)
        return frame, int(frame.size)
    if mode in ('fullframe_selective4', 'roi_only_selective4'):
        return render_selected(gray, tiles, chosen, feather, halo,
                               roi_only=(mode == 'roi_only_selective4'))
    raise ValueError(f'Unknown mode: {mode}')


def cpu_prep_bench(gray, tiles, chosen, modes, feather, halo, repeats, seed):
    """Warm and alternate the mode order; exclude read, writes, predictor, YOLO."""
    for mode in modes:
        make_frame(gray,tiles,chosen,mode,feather,halo)
    rng = random.Random(seed)
    times = {mode: [] for mode in modes}
    for _ in range(repeats):
        order = list(modes)
        rng.shuffle(order)
        for mode in order:
            begin = time.perf_counter_ns()
            _, _ = make_frame(gray,tiles,chosen,mode,feather,halo)
            times[mode].append((time.perf_counter_ns() - begin)/1e6)
    return {k: statistics.median(v) for k,v in times.items()}


def summarize(rows):
    result = {}
    for mode in MODES:
        r = [item for item in rows if item['method'] == mode]
        if not r:
            raise ValueError(f'No records for {mode}')
        result[mode] = {
            'images':len(r),
            'mean_cpu_preparation_ms_median_per_image': round(statistics.mean(x['cpu_prep_ms'] for x in r),4),
            'mean_detector_ms_per_image': round(statistics.mean(x['detect_ms'] for x in r),4),
            'mean_selector_ms_per_image': round(statistics.mean(x['selector_ms'] for x in r),4),
            'mean_sequential_prepare_select_detect_ms_per_image': round(statistics.mean(x['measured_pipeline_ms'] for x in r),4),
            'mean_interpolated_cubic_fraction_with_halo': round(statistics.mean(x['cubic_pixels']/(640*512) for x in r),5),
            'small_tp':sum(x['small_tp'] for x in r),
            'small_gt':sum(x['small_gt'] for x in r),
            'false_positives':sum(x['fp'] for x in r),
        }
        gt = result[mode]['small_gt']
        result[mode]['small_recall'] = round(result[mode]['small_tp']/gt,6) if gt else None
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path('configs/selective_sr_eval.yaml'))
    ap.add_argument('--limit',type=int,default=4,help='Step 5B validation sequences; 0=all')
    ap.add_argument('--out',type=Path,default=Path('reports/utility/roi_compute_smoke_v1'))
    ap.add_argument('--repeats',type=int,default=20,help='CPU preparation repeats per image')
    ap.add_argument('--halo',type=int,default=2,help='LR pixels surrounding each ROI; >=2')
    args=ap.parse_args()
    if not args.config.is_file() or args.limit<0 or not (3<=args.repeats<=200) or args.halo<2:
        ap.error('Require existing config, nonnegative limit, 3<=repeats<=200 and halo>=2')
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not isinstance(cfg,dict) or REQUIRED-set(cfg):
        ap.error('Incomplete Step 6A config')
    if int(cfg['budget_tiles'])!=4 or int(cfg['feather_hr_px'])!=8 or int(cfg['imgsz'])!=640:
        ap.error('Use Step 6A-compatible budget=4, feather=8, imgsz=640')
    if float(cfg['confidence'])!=0.25 or float(cfg['nms_iou'])!=0.7:
        ap.error('Use Step 6A fixed conf=.25 and NMS IoU=.7')
    if str(cfg['device'])!='0' or not torch.cuda.is_available():
        ap.error('CUDA device 0 is required to reproduce the prior GPU pilot')
    lr_root=Path(cfg['train_lr'])
    if lr_root.name!='train' or lr_root.parent.name!='images' or 'yolo_flir_lr_x2' not in lr_root.parts:
        ap.error('Require Step 6A TRAIN domain image source')
    if args.out.exists():
        ap.error('Output already exists: choose a new --out')
    paths=get_eval_image_names(Path(cfg['utility_csv']),Path(cfg['split_manifest']),args.limit)
    for key in ('predictor_checkpoint','detector_checkpoint'):
        if not Path(cfg[key]).is_file():
            ap.error(f'Missing checkpoint: {cfg[key]}')
    if not lr_root.is_dir():
        ap.error('Missing LR train images directory')
    labels=lr_root.parent.parent/'labels'/'train'
    if not labels.is_dir():
        ap.error('Missing YOLO train labels directory')
    device=torch.device('cuda:0')
    model,input_size=predictor_model(Path(cfg['predictor_checkpoint']),Path(cfg['utility_csv']),device)
    from ultralytics import YOLO
    detector=YOLO(str(cfg['detector_checkpoint']))
    cv2.setNumThreads(1)  # reproducible CPU preparation settings, not GPU control
    tiles=roi_grid(640,512,4,4)
    warm=np.zeros(LR_SHAPE,np.uint8)
    predict_tile_scores(model,warm,tiles,input_size,device,float(cfg['false_positive_penalty']))
    prediction_records(detector,cv2.resize(warm,HR_SIZE),cfg)
    torch.cuda.synchronize(device)
    args.out.mkdir(parents=True)
    rows=[]
    for i,name in enumerate(paths,1):
        gray=cv2.imread(str(lr_root/name),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=LR_SHAPE:
            raise ValueError(f'Invalid LR input {name}')
        label=labels/(Path(name).stem+'.txt')
        if not label.is_file():
            raise FileNotFoundError(label)
        # Selection inputs are ONLY grayscale LR and tile coordinates, never GT.
        torch.cuda.synchronize(device)
        start=time.perf_counter()
        scores=predict_tile_scores(model,gray,tiles,input_size,device,float(cfg['false_positive_penalty']))
        chosen=choose_tiles('predictor4',name,4,len(tiles),int(cfg['seed']),scores)
        torch.cuda.synchronize(device)
        selector_ms=(time.perf_counter()-start)*1000
        expected,_=make_frame(gray,tiles,chosen,'fullframe_selective4',8,args.halo)
        actual,processed=make_frame(gray,tiles,chosen,'roi_only_selective4',8,args.halo)
        maxdiff=int(np.max(np.abs(expected.astype(np.int16)-actual.astype(np.int16))))
        differing=int(np.count_nonzero(expected!=actual))
        if differing:
            raise RuntimeError(f'ROI interpolation mismatch in {name}: {differing} pixels, max abs diff {maxdiff}')
        times=cpu_prep_bench(gray,tiles,chosen,MODES,8,args.halo,args.repeats,int(cfg['seed'])+i)
        gts=ground_truth(label) # strictly AFTER selection
        # Rotate detector evaluation order to counter systematic warmup/order bias.
        order=list(MODES)
        random.Random(int(cfg['seed'])+i).shuffle(order)
        for mode in order:
            torch.cuda.synchronize(device)
            begin=time.perf_counter()
            frame,cubic_pixels=make_frame(gray,tiles,chosen,mode,8,args.halo)
            torch.cuda.synchronize(device)
            middle=time.perf_counter()
            preds=prediction_records(detector,frame,cfg)
            torch.cuda.synchronize(device)
            end=time.perf_counter()
            cnt=detection_counts(gts,preds,float(cfg['confidence']))
            rows.append({
                'image':name,'sequence':Path(name).stem.split('-frame-')[0],
                'method':mode,'selected_tiles':' '.join(map(str,chosen)),
                'small_tp':cnt['small_tp'],'small_gt':cnt['small_gt'],'fp':cnt['fp'],
                'cpu_prep_ms':round(times[mode],5),
                'selector_ms':round(selector_ms if 'selective' in mode else 0.,5),
                'detect_ms':round((end-middle)*1000,5),
                'measured_pipeline_ms':round((end-begin)*1000 + (selector_ms if 'selective' in mode else 0),5),
                'cubic_pixels':cubic_pixels,
                'exact_match': (mode not in ('fullframe_selective4','roi_only_selective4') or differing==0),
            })
        print(f'Checked {i}/{len(paths)} images | ROI vs matched full-frame pixels: EXACT | cropped cubic: {processed/(640*512):.1%} of full output',flush=True)
    # Two selective conditions must be identical on every frame, not just totals.
    grouped={}
    for r in rows:
        grouped[(r['image'],r['method'])]=r
    for name in paths:
        a=grouped[(name,'fullframe_selective4')]
        b=grouped[(name,'roi_only_selective4')]
        if (a['small_tp'],a['small_gt'],a['fp'])!=(b['small_tp'],b['small_gt'],b['fp']):
            raise RuntimeError(f'Selective YOLO diagnostic mismatch despite exact image pixels: {name}')
    with (args.out/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary={
        'task':'UtiTherm-SR Step 6B true ROI-only OpenCV Bicubic pilot',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'images':len(paths),'repeats_per_frame':args.repeats,
        'roi_halo_lr_px':args.halo, 'cpu_opencv_threads':cv2.getNumThreads(),
        'checkpoint_detector':str(cfg['detector_checkpoint']),
        'checkpoint_predictor':str(cfg['predictor_checkpoint']),
        'split':'Step 5B internal validation; detector TRAIN domain',
        'selected_budget_tiles':4,'enhanced_area_fraction':0.25,
        'exact_pixel_and_diagnostic_equivalence':True,
        'results':summarize(rows),
        'limitations':[
            'Matched OpenCV cubic full-frame-vs-ROI implementations use same blending; exactly equivalent pixels.',
            'Whole-frame linear interpolation remains necessary; only cubic is cropped.',
            'Cropped cubic includes a 2-LR-pixel halo, so cubic output pixels exceed enhanced image area.',
            'Bicubic is very cheap; ROI-only may be slower due to multiple resize calls and selector cost.',
            'GPU speed uses one sequential prediction per mode per image and excludes file I/O; repeat on more frames for timing claims.',
            'The predictor is selected on its own validation subset; YOLO already trained on these detector TRAIN frames.',
            'No independent detection test; do not claim novelty or compute savings without evidence.',
            'Applies to OpenCV cubic interpolation, not learned FSRCNN ROI enhancement.'
        ]
    }
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('\nMEASURED SUMMARY (ms/image; smaller is better):')
    for mode,r in summary['results'].items():
        print(f"  {mode:23s} CPU prep {r['mean_cpu_preparation_ms_median_per_image']:.3f} | "
              f"selector {r['mean_selector_ms_per_image']:.3f} | "
              f"pipeline {r['mean_sequential_prepare_select_detect_ms_per_image']:.3f} | "
              f"smallTP {r['small_tp']} | FP {r['false_positives']}")
    print(f'DONE — {args.out / "summary.json"} ; {args.out / "per_image.csv"}',flush=True)


if __name__=='__main__':
    main()
