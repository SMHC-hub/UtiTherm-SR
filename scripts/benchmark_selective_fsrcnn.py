"""UtiTherm-SR Step 6C2: true selective FSRCNN x2, detection and GPU benchmarking.

Run inside the EXISTING Docker research service. Same 20 predictor-held-out
TRAIN-domain frames as Steps 6A-6C1. Never tune on official FLIR val/test.

    python scripts/benchmark_selective_fsrcnn.py --limit 4 --repeats 3 \
      --out reports/utility/fsrcnn_selective_smoke_v1

Five policies use one fixed detector checkpoint. The paired selective conditions
use the SAME chosen four tiles; only full-frame vs crop neural computation differs.
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
sys.path.insert(0, str(ROOT / 'src'))

from utitherm.sr.fsrcnn import FSRCNN
from utitherm.sr.roi_fsrcnn import render_fsrcnn, selected_region_difference
from evaluate_selective_sr import (
    choose_tiles, get_eval_image_names, predict_tile_scores, predictor_model,
)
from probe_roi_utility import roi_grid, prediction_records, detection_counts
from evaluate_bicubic_baseline import ground_truth

MODES = ('linear_640','full_bicubic_640','full_fsrcnn_640',
         'selective_fullframe_fsrcnn4','selective_roi_fsrcnn4')
GRID = (640, 512)
LR_SHAPE = (256, 320)


def validate_settings(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError('Require Step 6A dictionary config')
    required = {'train_lr','utility_csv','split_manifest','predictor_checkpoint',
                'detector_checkpoint','budget_tiles','seed','imgsz','confidence',
                'nms_iou','false_positive_penalty','feather_hr_px','device'}
    if required - set(cfg):
        raise ValueError(f'Missing config fields: {sorted(required-set(cfg))}')
    if int(cfg['budget_tiles']) != 4 or int(cfg['feather_hr_px']) != 8:
        raise ValueError('Keep budget=4 and feather=8 as Step 6A')
    if int(cfg['imgsz']) != 640:
        raise ValueError('YOLO input must be imgsz 640')
    if str(cfg['device']) != '0':
        raise ValueError('Use CUDA GPU device 0')
    if abs(float(cfg['confidence']) - .25) > 1e-8 or abs(float(cfg['nms_iou']) - .7) > 1e-8:
        raise ValueError('Use Step 6A fixed conf=.25 and NMS IoU=.7')
    if abs(float(cfg['false_positive_penalty']) - .25) > 1e-8:
        raise ValueError('Use fixed false-positive penalty=.25')
    lr_root = Path(cfg['train_lr'])
    if lr_root.name != 'train' or lr_root.parent.name != 'images' or 'yolo_flir_lr_x2' not in lr_root.parts:
        raise ValueError('Safety guard: only official detector TRAIN-domain LR images')


def load_sr_checkpoint(path: Path, device: torch.device) -> FSRCNN:
    if not path.is_file():
        raise FileNotFoundError(f'Missing FSRCNN checkpoint: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or 'state_dict' not in payload or 'model_kwargs' not in payload:
        raise ValueError('Require the Step 4B trained FSRCNN state_dict/model_kwargs checkpoint')
    kw = payload['model_kwargs']
    if not isinstance(kw, dict) or kw.get('scale') != 2:
        raise ValueError('Expected FSRCNN x2 model')
    model = FSRCNN(**kw)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.to(device).eval()


def render(mode: str, gray: np.ndarray, tiles: list, chosen: list[int],
           sr: FSRCNN, device: torch.device, halo: int) -> tuple[np.ndarray, int]:
    if mode == 'full_bicubic_640':
        return cv2.resize(gray, GRID, interpolation=cv2.INTER_CUBIC), 0
    m = {'linear_640':'linear','full_fsrcnn_640':'full',
         'selective_fullframe_fsrcnn4':'selective_fullframe',
         'selective_roi_fsrcnn4':'selective_roi'}.get(mode)
    if m is None:
        raise ValueError(f'Unknown benchmark policy {mode}')
    return render_fsrcnn(gray, tiles, chosen, sr, device, m, feather=8, halo=halo)


def time_trial(mode: str, gray: np.ndarray, name: str, tiles: list,
               predictor, input_size: tuple[int,int], sr: FSRCNN,
               detector, cfg: dict, device: torch.device,
               expected: tuple[int,...], halo: int) -> tuple[dict, list[dict]]:
    """Measure SELECT->SR->YOLO, synchronizing all GPU work at stage boundaries."""
    if mode not in MODES:
        raise ValueError(f'Unknown benchmark policy {mode}')
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0=time.perf_counter_ns()
    chosen=[]
    if mode.startswith('selective_'):
        scores = predict_tile_scores(predictor, gray, tiles, input_size,
                                     device, float(cfg['false_positive_penalty']))
        chosen = choose_tiles('predictor4', name, 4, len(tiles), int(cfg['seed']), scores)
        if tuple(chosen) != expected:
            raise RuntimeError('Predictor rankings changed during repeated benchmark trials')
        torch.cuda.synchronize(device)
    t1=time.perf_counter_ns()
    frame, computed_pixels = render(mode, gray, tiles, chosen, sr, device, halo)
    torch.cuda.synchronize(device)
    t2=time.perf_counter_ns()
    preds=prediction_records(detector, frame, cfg)
    torch.cuda.synchronize(device)
    t3=time.perf_counter_ns()
    peak_mib=torch.cuda.max_memory_allocated(device)/(1024**2)
    return {
        'selector_ms':(t1-t0)/1e6, 'sr_or_resize_ms':(t2-t1)/1e6,
        'detector_ms':(t3-t2)/1e6, 'total_ms':(t3-t0)/1e6,
        'model_input_lr_pixels':computed_pixels,
        'pytorch_peak_allocated_mib':peak_mib,
    },preds


def collapse_and_compare(rows: list[dict], seed: int = 42,
                         resamples: int = 1000) -> dict:
    """Repeated trials are grouped by image, not mistaken for independent scenes."""
    if not rows or resamples < 100:
        raise ValueError('Need rows and >=100 bootstrap samples')
    images=sorted({row['image'] for row in rows})
    repeats=sorted({int(row['repeat']) for row in rows})
    if len(rows) != len(images)*len(repeats)*len(MODES):
        raise ValueError('Incomplete or duplicate benchmark trial count')
    measures=('selector_ms','sr_or_resize_ms','detector_ms','total_ms','pytorch_peak_allocated_mib')
    grouped=defaultdict(list)
    for row in rows:
        key=(row['image'],row['method'],int(row['repeat']))
        if key in grouped:
            raise ValueError('Duplicate image/mode/repeat row')
        if row['method'] not in MODES:
            raise ValueError('Unknown method in trial rows')
        grouped[key]=[float(row[k]) for k in measures]
    summary={}
    meds={}
    for mode in MODES:
        values={field:[] for field in measures}
        for im in images:
            by_repeat=[grouped[(im,mode,r)] for r in repeats]
            for k,field in enumerate(measures):
                values[field].append(float(statistics.median(x[k] for x in by_repeat)))
        meds[mode]=values['total_ms']
        summary[mode]={k:{'mean_of_per_image_medians':round(statistics.mean(v),4),
                          'median_of_per_image_medians':round(statistics.median(v),4)}
                       for k,v in values.items()}
    rng=np.random.default_rng(seed)
    differences={}
    for comparator in ('full_fsrcnn_640','selective_fullframe_fsrcnn4','full_bicubic_640'):
        d=np.asarray(meds[comparator])-np.asarray(meds['selective_roi_fsrcnn4'])
        boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(resamples)])
        differences[f'{comparator}_minus_selective_roi_fsrcnn4_ms']={
            'mean_paired_difference_ms':round(float(d.mean()),4),
            'exploratory_image_bootstrap_95ci':[round(float(np.percentile(boot,2.5)),4),
                                                 round(float(np.percentile(boot,97.5)),4)],
            'positive_means_roi_is_faster':True,
        }
    return {'images':len(images),'repeats':len(repeats),'methods':summary,
            'paired_comparisons':differences}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('Cannot write empty records')
    with path.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/selective_sr_eval.yaml'))
    p.add_argument('--sr-checkpoint',type=Path,default=Path('models/sr/fsrcnn_x2_v1.pt'))
    p.add_argument('--limit',type=int,default=4,help='Predictor-held-out TRAIN-domain images; 0=all 20')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--halo',type=int,default=16,help='LR context halo; may increase only after equivalence tests')
    p.add_argument('--max-diff',type=int,default=1,help='Maximum allowed 8-bit pixel difference in matched selective outputs')
    p.add_argument('--out',type=Path,default=Path('reports/utility/fsrcnn_selective_smoke_v1'))
    a=p.parse_args()
    if a.limit<0 or not 3<=a.repeats<=20 or not 2<=a.warmup<=20 or a.halo<10 or not 0<=a.max_diff<=2:
        p.error('Require limit>=0, repeats=3..20, warmup=2..20, halo>=10, max-diff=0..2')
    if a.out.exists():
        p.error(f'Results directory exists: {a.out}; choose a new --out')
    if not a.config.is_file():
        p.error(f'Config missing: {a.config}')
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    try:
        validate_settings(cfg)
        lr_root=Path(cfg['train_lr'])
        if not lr_root.is_dir():
            raise FileNotFoundError(lr_root)
        for k in ('utility_csv','split_manifest','predictor_checkpoint','detector_checkpoint'):
            if not Path(cfg[k]).is_file():
                raise FileNotFoundError(cfg[k])
        names=get_eval_image_names(Path(cfg['utility_csv']),Path(cfg['split_manifest']),a.limit)
        if not names:
            raise ValueError('No eligible predictor-held-out TRAIN images')
        if not a.sr_checkpoint.is_file():
            raise FileNotFoundError(a.sr_checkpoint)
        label_root=lr_root.parent.parent/'labels'/'train'
        if not label_root.is_dir():
            raise FileNotFoundError(label_root)
    except (ValueError,FileNotFoundError,KeyError) as exc:
        p.error(str(exc))
    if not torch.cuda.is_available():
        p.error('Step 6C2 benchmark requires the NVIDIA CUDA GPU')
    device=torch.device('cuda:0')
    cv2.setNumThreads(1)
    predictor,input_size=predictor_model(Path(cfg['predictor_checkpoint']),Path(cfg['utility_csv']),device)
    sr=load_sr_checkpoint(a.sr_checkpoint,device)
    from ultralytics import YOLO
    detector=YOLO(str(cfg['detector_checkpoint']))
    tiles=roi_grid(640,512,4,4)
    warm=np.zeros(LR_SHAPE,np.uint8)
    for _ in range(a.warmup):
        predict_tile_scores(predictor,warm,tiles,input_size,device,float(cfg['false_positive_penalty']))
        for mode in MODES:
            frame,_=render(mode,warm,tiles,[0,5,10,15],sr,device,a.halo)
            prediction_records(detector,frame,cfg)
    torch.cuda.synchronize(device)
    rows=[]; diag=[]; pixel_audits=[]
    # Only the predictor's internal sequence-held-out set; NEVER held-out detector VAL for tuning.
    for i,name in enumerate(names,1):
        gray=cv2.imread(str(lr_root/name),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=LR_SHAPE:
            raise ValueError(f'Bad LR frame {name}; expected 320x256 grayscale')
        label=label_root/(Path(name).stem+'.txt')
        if not label.is_file():
            raise FileNotFoundError(label)
        sc=predict_tile_scores(predictor,gray,tiles,input_size,device,float(cfg['false_positive_penalty']))
        chosen=tuple(choose_tiles('predictor4',name,4,16,int(cfg['seed']),sc))
        ref,npix_full=render('selective_fullframe_fsrcnn4',gray,tiles,list(chosen),sr,device,a.halo)
        crop,npix_crop=render('selective_roi_fsrcnn4',gray,tiles,list(chosen),sr,device,a.halo)
        audit=selected_region_difference(ref,crop,tiles,list(chosen))
        if audit['max_abs_difference']>a.max_diff:
            raise RuntimeError(f'ROI neural boundary mismatch {name}: {audit}; '
                               'inspect the FSRCNN halo and quantization before benchmarking')
        pixel_audits.append({'image':name,'selected_tiles':' '.join(map(str,chosen)),
                             'model_input_fraction_with_halo':round(npix_crop/npix_full,6),**audit})
        groundtruth=None
        for rep in range(a.repeats):
            order=list(MODES)
            random.Random(int(cfg['seed'])+i*1009+rep).shuffle(order)
            for mode in order:
                measure,preds=time_trial(mode,gray,name,tiles,predictor,input_size,sr,
                                         detector,cfg,device,chosen,a.halo)
                rows.append({'image':name,'repeat':rep,'method':mode,
                             **{k:(round(v,6) if isinstance(v,float) else v)
                                for k,v in measure.items()}})
                if rep==0:
                    if groundtruth is None:
                        groundtruth=ground_truth(label)
                    counts=detection_counts(groundtruth,preds,float(cfg['confidence']))
                    diag.append({'image':name,'method':mode,**counts})
        print(f'Compared {i}/{len(names)} frames | full-vs-ROI max pixel diff={audit["max_abs_difference"]} | '
              f'FSRCNN cropped input {npix_crop/npix_full:.1%}',flush=True)
    timing=collapse_and_compare(rows,seed=int(cfg['seed']))
    detection={}
    for mode in MODES:
        rs=[r for r in diag if r['method']==mode]
        counts={k:sum(r[k] for r in rs) for k in ('small_gt','small_tp','all_tp','fp')}
        detection[mode]={**counts,'small_recall_conf_025':round(counts['small_tp']/counts['small_gt'],6)
                         if counts['small_gt'] else None}
    a.out.mkdir(parents=True)
    write_csv(a.out/'timing_per_trial.csv',rows)
    write_csv(a.out/'detection_per_image.csv',diag)
    write_csv(a.out/'roi_equivalence.csv',pixel_audits)
    summary={
        'task':'Step 6C2 cropped neural SR vs full FSRCNN x2 and detector controls',
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'images':len(names),'repeats':a.repeats,'halo_lr_px':a.halo,
        'allowed_max_roi_difference_u8':a.max_diff,
        'max_observed_roi_difference_u8':max(r['max_abs_difference'] for r in pixel_audits),
        'mean_cropped_model_input_fraction':round(statistics.mean(r['model_input_fraction_with_halo'] for r in pixel_audits),6),
        'checkpoints':{
            k:{'path':str(pth),'sha256':hashlib.sha256(pth.read_bytes()).hexdigest()}
            for k,pth in {
                'sr':a.sr_checkpoint,'detector':Path(cfg['detector_checkpoint']),
                'predictor':Path(cfg['predictor_checkpoint'])}.items()
        },
        'source':'Step 5B sequence-held-out subset of DETECTOR TRAIN images',
        'timing_excludes':'disk image loading, writing outputs and annotation processing',
        'timings_ms':timing,
        'detection':detection,
        'warnings':[
            'FSRCNN tiles are selected by a predictor trained on *Bicubic* marginal utility; selection transfer is unproven.',
            'Pilot images belong to the already-trained YOLO detector TRAIN domain, not an independent detector test.',
            'Predictor checkpoint was selected using the internal validation domain; pilot is NOT an independent test.',
            'ROI crop uses context halo and may differ by 1-2 uint8 levels due to numeric kernels; report the measured difference.',
            'Fewer model-input pixels do not automatically imply fewer measured FLOPs, GPU energy, VRAM or lower end-to-end latency.',
            'GPU peak allocation measures PyTorch memory allocation, not total GPU use or power.',
            'No official COCO AP in this script; only class-aware diagnostic TP/FP at conf=.25 and IoU=.5.',
            'Per-image medians used for timing; bootstrap uncertainty is exploratory with correlated scenes.',
            'FSRCNN trained for synthetic degradation; real-world sensor fidelity and scientific novelty are unproven.',
        ]
    }
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('\nSTEP 6C2 RESULTS — medians over randomized paired repeats (ms/image):')
    for mode in MODES:
        measures=timing['methods'][mode]
        counts=detection[mode]
        print(f'  {mode:31s} total={measures["total_ms"]["mean_of_per_image_medians"]:7.3f} '
              f'predict={measures["selector_ms"]["mean_of_per_image_medians"]:6.3f} '
              f'SR/resize={measures["sr_or_resize_ms"]["mean_of_per_image_medians"]:6.3f} '
              f'smallTP={counts["small_tp"]} FP={counts["fp"]}')
    print(f'Max ROI pixel mismatch={summary["max_observed_roi_difference_u8"]}; '
          f'mean cropped input fraction={summary["mean_cropped_model_input_fraction"]:.1%}')
    print(f'DONE — {a.out / "summary.json"} ; {a.out / "timing_per_trial.csv"}',flush=True)


if __name__=='__main__':
    main()
