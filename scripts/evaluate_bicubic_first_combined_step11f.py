"""Step 11F: ACTUAL combined four-tile Large FSRCNN over full Bicubic.

Read-only 200-image frozen FLIR validation DEVELOPMENT study (NOT final test).
Same-run controls: full Bicubic, full Large FSRCNN, random four Bicubic-first,
and cheap tile-standard-deviation contrast top-four Bicubic-first.

Usage, at repository root in research Docker:
  python scripts/evaluate_bicubic_first_combined_step11f.py --limit 8 --repeats 3 --out reports/development/step11f_smoke_8_v1
  python scripts/evaluate_bicubic_first_combined_step11f.py --limit 0 --repeats 3 --out reports/development/step11f_full_200_v1
"""
from __future__ import annotations

import argparse
import csv
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

MODES = (
    'full_bicubic_640',
    'full_large_fsrcnn_640',
    'random4_bicubic_first_h10',
    'contrast4_bicubic_first_h10',
)
COMPARISONS = (
    ('contrast4_bicubic_first_h10','full_bicubic_640'),
    ('random4_bicubic_first_h10','full_bicubic_640'),
    ('contrast4_bicubic_first_h10','random4_bicubic_first_h10'),
    ('contrast4_bicubic_first_h10','full_large_fsrcnn_640'),
    ('full_large_fsrcnn_640','full_bicubic_640'),
)
SHAPE = (256,320)
HR = (640,512)


def validate_protocol(exp:dict, base:dict, manifest:dict) -> None:
    """Reject a silent change to a measured detector or enhancement protocol."""
    if not all(isinstance(d,dict) for d in (exp,base,manifest)):
        raise ValueError('Expected experiment, base, manifest mappings')
    for key,value in (('expected_images',200),('expected_sequences',17),
                      ('budget_tiles',4),('halo',10),('feather_hr_px',8),
                      ('false_positive_penalty',.25),('confidence',.25),('nms_iou',.7)):
        if float(exp.get(key,-1)) != float(value):
            raise ValueError(f'Step11F protocol changed: {key}')
    if list(exp.get('methods',[])) != list(MODES):
        raise ValueError('Methods/order changed from Step11F four-arm protocol')
    for key,value in (('imgsz',640),('budget_tiles',4),('feather_hr_px',8),
                      ('false_positive_penalty',.25),('confidence',.25),('nms_iou',.7)):
        if float(base.get(key,-1)) != float(value):
            raise ValueError(f'Frozen detection config changed: {key}')
    if str(base.get('device')) != '0':
        raise ValueError('Require CUDA device 0')
    if manifest.get('protocol') != 'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST':
        raise ValueError('Unexpected split protocol')
    if (int(manifest.get('selected_images',-1)) != 200 or
            int(manifest.get('unique_sequences',-1)) != 17):
        raise ValueError('Frozen 200-frame/17-sequence manifest changed')


def contrast_scores(lr:np.ndarray, tiles:list[tuple[int,int,int,int]]) -> np.ndarray:
    """Same tile_std feature as Step11D; faster than extracting 20 unused features.

    Step11D computes tile_std using np.float32 LR pixels and .std() at each
    HR/2 window. Selection never examines GT, predictions, or measured utility.
    """
    if lr.shape != SHAPE or lr.dtype != np.uint8 or lr.ndim != 2:
        raise ValueError('Expected 320x256 single-channel uint8 LR frame')
    if len(tiles)!=16:
        raise ValueError('Expected exactly sixteen 4x4 tiles')
    f32 = lr.astype(np.float32)
    scores = []
    for x0,y0,x1,y1 in tiles:
        if not (0<=x0<x1<=640 and 0<=y0<y1<=512) or any(v%2 for v in (x0,y0,x1,y1)):
            raise ValueError('Invalid tile rectangle')
        scores.append(float(f32[y0//2:y1//2,x0//2:x1//2].std()))
    result = np.asarray(scores,dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite contrast scores')
    return result


def topk(scores:np.ndarray, k:int) -> list[int]:
    scores = np.asarray(scores,dtype=np.float64)
    if scores.shape != (16,) or not np.isfinite(scores).all() or not 0<=k<=16:
        raise ValueError('Invalid ranking scores/budget')
    return sorted(range(16),key=lambda i:(-float(scores[i]),i))[:k]


def selected_tiles(method:str, lr:np.ndarray, name:str, tiles:list, seed:int,
                   random_tile_fn) -> list[int]:
    if method in ('full_bicubic_640','full_large_fsrcnn_640'):
        return []
    if method == 'random4_bicubic_first_h10':
        indices = random_tile_fn('random4',name,4,16,seed)
    elif method == 'contrast4_bicubic_first_h10':
        indices = topk(contrast_scores(lr,tiles),4)
    else:
        raise ValueError(f'Unknown method: {method}')
    if len(indices) != 4 or len(set(indices)) != 4 or any(not 0<=i<16 for i in indices):
        raise ValueError('Selection must contain four distinct tiles')
    return list(indices)


def render(method:str, lr:np.ndarray, tiles:list, selected:list[int], sr, device,
           combined_renderer, full_renderer) -> tuple[np.ndarray,int]:
    if method == 'full_bicubic_640':
        return cv2.resize(lr,HR,interpolation=cv2.INTER_CUBIC),0
    if method == 'full_large_fsrcnn_640':
        return full_renderer(lr,tiles,[],sr,device,'full')
    if method in ('random4_bicubic_first_h10','contrast4_bicubic_first_h10'):
        if len(selected)!=4 or len(set(selected))!=4:
            raise ValueError('Combined method requires four distinct tiles')
        # IMPORTANT: Step11A renderer composites on *full Bicubic*, not linear.
        return combined_renderer(lr,tiles,selected,sr,device,feather=8,halo=10)
    raise ValueError(f'Unknown method {method}')


def synchronize(device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def timed_trial(method:str, lr:np.ndarray, image:str, tiles:list, seed:int,
                sr,detector,base:dict,device,random_tile_fn,
                combined_renderer,full_renderer,predict_fn,
                frozen_selection:dict) -> tuple[dict,list]:
    synchronize(device)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0=time.perf_counter_ns()
    selected=selected_tiles(method,lr,image,tiles,seed,random_tile_fn)
    if tuple(selected) != tuple(frozen_selection[method]):
        raise RuntimeError(f'Tile selection changed across repeats for {image}, {method}')
    synchronize(device)
    t1=time.perf_counter_ns()
    enhanced,pixels=render(method,lr,tiles,selected,sr,device,combined_renderer,full_renderer)
    if enhanced.shape != (512,640) or enhanced.dtype != np.uint8:
        raise RuntimeError('Rendered image has incorrect format')
    if pixels<0:
        raise RuntimeError('Invalid processed pixel count')
    synchronize(device)
    t2=time.perf_counter_ns()
    preds=predict_fn(detector,enhanced,base)
    synchronize(device)
    t3=time.perf_counter_ns()
    return ({'selector_ms':(t1-t0)/1e6,'sr_ms':(t2-t1)/1e6,
             'detector_ms':(t3-t2)/1e6,'total_ms':(t3-t0)/1e6,
             'computed_pixels':int(pixels),
             'peak_torch_mib': (torch.cuda.max_memory_allocated(device)/1024**2
                                if device.type=='cuda' else 0.0)},preds)


def cluster_ci(deltas:dict[str,float], seq_map:dict[str,list[str]],
               seed:int, num_samples:int) -> list[float]:
    if len(seq_map)<2 or num_samples<100 or set(deltas)!={i for ims in seq_map.values() for i in ims}:
        raise ValueError('Need complete image/sequence membership and >=100 bootstraps')
    keys=sorted(seq_map)
    sums=np.array([sum(deltas[i] for i in seq_map[s]) for s in keys],dtype=np.float64)
    sizes=np.array([len(seq_map[s]) for s in keys],dtype=np.float64)
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(keys),size=(num_samples,len(keys)))
    vals=sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)
    return [round(float(x),6) for x in np.quantile(vals,[.025,.975])]


def summarize(detections:list[dict],trials:list[dict],n_repeats:int,
              fp_penalty:float,seed:int,bootstrap_iterations:int) -> dict:
    if not detections or not trials or n_repeats<3:
        raise ValueError('Missing detection/timing rows or invalid repeats')
    by_image=defaultdict(dict)
    for r in detections:
        im,m=r['image'],r['method']
        if m not in MODES or m in by_image[im] or r['sequence']!=seq_of(im):
            raise ValueError('Invalid/duplicate detection record')
        by_image[im][m]=r
    names=sorted(by_image)
    if any(set(x)!=set(MODES) for x in by_image.values()):
        raise ValueError('Incomplete methods in detection CSV')
    for im in names:
        if len({int(by_image[im][m]['small_gt']) for m in MODES}) != 1:
            raise ValueError('Ground-truth counts disagree between methods')
    by_seq=defaultdict(list)
    for im in names:
        by_seq[seq_of(im)].append(im)
    if len(by_seq)<2:
        raise ValueError('Need multiple video sequences')
    timings={}
    for t in trials:
        key=(t['image'],t['method'],int(t['repeat']))
        if (key in timings or t['image'] not in by_image or t['method'] not in MODES or
                t['sequence']!=seq_of(t['image']) or not 0<=int(t['repeat'])<n_repeats):
            raise ValueError('Unexpected or duplicate timing trial')
        timings[key]=t
    if len(timings)!=len(names)*len(MODES)*n_repeats:
        raise ValueError('Missing or extra timing trials')
    for im in names:
        for m in MODES:
            for j in range(n_repeats):
                if (im,m,j) not in timings:
                    raise ValueError('Missing timing row')
    results={}; times={}
    for m in MODES:
        counts={key:sum(int(by_image[im][m][key]) for im in names)
                for key in ('small_gt','small_tp','all_tp','fp')}
        met={im:{key:statistics.median(float(timings[im,m,j][key]) for j in range(n_repeats))
                  for key in ('selector_ms','sr_ms','detector_ms','total_ms','computed_pixels','peak_torch_mib')}
             for im in names}
        times[m]=met
        results[m]={**counts,
                    'small_recall_conf_025':round(counts['small_tp']/counts['small_gt'],6)
                           if counts['small_gt'] else None,
                    'mean_image_median_total_ms':round(statistics.mean(v['total_ms'] for v in met.values()),6),
                    'mean_selector_ms':round(statistics.mean(v['selector_ms'] for v in met.values()),6),
                    'mean_sr_ms':round(statistics.mean(v['sr_ms'] for v in met.values()),6),
                    'mean_detector_ms':round(statistics.mean(v['detector_ms'] for v in met.values()),6),
                    'mean_processed_lr_pixels':round(statistics.mean(v['computed_pixels'] for v in met.values()),2),
                    'mean_peak_torch_mib':round(statistics.mean(v['peak_torch_mib'] for v in met.values()),3)}
    comparisons={}
    for ix,(a,b) in enumerate(COMPARISONS):
        u={im:(int(by_image[im][a]['small_tp'])-int(by_image[im][b]['small_tp']))
                -fp_penalty*(int(by_image[im][a]['fp'])-int(by_image[im][b]['fp'])) for im in names}
        t={im:times[a][im]['total_ms']-times[b][im]['total_ms'] for im in names}
        comparisons[f'{a}__vs__{b}']={
            'delta_small_tp':results[a]['small_tp']-results[b]['small_tp'],
            'delta_fp':results[a]['fp']-results[b]['fp'],
            'delta_utility_total':round(sum(u.values()),6),
            'delta_utility_per_image':round(statistics.mean(u.values()),6),
            'sequence_bootstrap_utility_per_image_95ci':cluster_ci(u,by_seq,seed+ix*31,bootstrap_iterations),
            'delta_latency_ms':round(statistics.mean(t.values()),6),
            'sequence_bootstrap_latency_ms_95ci':cluster_ci(t,by_seq,seed+ix*31+11,bootstrap_iterations),
            'image_utility_wins_ties_losses':{'wins':sum(v>0 for v in u.values()),
                                               'ties':sum(v==0 for v in u.values()),
                                               'losses':sum(v<0 for v in u.values())},
            'sequence_utility_wins_ties_losses':{'wins':sum(sum(u[im] for im in ims)>0 for ims in by_seq.values()),
                                                  'ties':sum(sum(u[im] for im in ims)==0 for ims in by_seq.values()),
                                                  'losses':sum(sum(u[im] for im in ims)<0 for ims in by_seq.values())},
        }
    return {'images':len(names),'sequences':len(by_seq),'repeats_per_image':n_repeats,
            'methods':results,'comparisons':comparisons,
            'utility_definition':f'delta small TP - {fp_penalty} * delta FP (conf=0.25, IoU=0.50)',
            'limitations':[
                'Reused frozen FLIR VAL development images already analyzed repeatedly; NOT independent final test.',
                'Diagnostics are class-aware small-object TP/FP at conf .25 IoU .50, not standard mAP.',
                'Actual FOUR-tile combined SR images, not retrospective single-tile sums.',
                'Single-seed random-four selection. Multiple random seeds needed for robust random baseline.',
                'Only 17 video clusters and exploratory confidence intervals.',
                'Runtime measured on GPU after warmup; excludes disk IO and checkpoint loading.',
                'Contrast top4 was selected after studying related train-domain data; no final novelty claim.',
            ]}


def seq_of(image:str) -> str:
    if '-frame-' not in image or not image.endswith('.png') or '/' in image or '\\' in image:
        raise ValueError(f'Unexpected video frame name: {image}')
    return image.split('-frame-',1)[0]


def reference_control_differences(detections:list[dict], reference_csv:Path, reference:dict) -> dict:
    """Match every per-image control against Step10D without discarding failed runs.

    This is independent of GPU timings or the four-region policies. A mismatch
    never becomes a pass merely because it is small.
    """
    controls=('full_bicubic_640','full_large_fsrcnn_640')
    fields=('small_gt','small_tp','all_tp','fp')
    if not reference_csv.is_file():
        raise FileNotFoundError(f'Step10D per-image reference is missing: {reference_csv}')
    with reference_csv.open('r',encoding='utf-8-sig',newline='') as fh:
        records=list(csv.DictReader(fh))
    previous={}
    for r in records:
        if r.get('method') not in controls:
            continue
        key=(r['image'],r['method'])
        if key in previous:
            raise ValueError(f'Duplicate reference control row: {key}')
        if r.get('sequence')!=seq_of(r['image']):
            raise ValueError(f'Invalid reference sequence for {key}')
        previous[key]=r
    current={}
    for r in detections:
        if r.get('method') not in controls:
            continue
        key=(r['image'],r['method'])
        if key in current:
            raise ValueError(f'Duplicate current control row: {key}')
        if r.get('sequence')!=seq_of(r['image']):
            raise ValueError(f'Invalid current sequence for {key}')
        current[key]=r
    if not current or set(current)!=set(previous):
        missing=sorted(set(previous)-set(current))[:5]
        extra=sorted(set(current)-set(previous))[:5]
        raise ValueError(f'Step10D control images differ: missing={missing}, extra={extra}')
    # Validate reference CSV against the official Step10D aggregate summary.
    for mode in controls:
        rows=[r for (_,m),r in previous.items() if m==mode]
        for field in fields:
            subtotal=sum(int(r[field]) for r in rows)
            expected=int(reference['results']['methods'][mode][field])
            if subtotal!=expected:
                raise ValueError(f'Step10D reference CSV disagrees with its summary: '
                                 f'{mode}.{field}: csv={subtotal}, summary={expected}')
    diffs=[]; counts={}
    for mode in controls:
        counts[mode]={'mismatch_images':0, 'delta_totals':{key:0 for key in fields}}
    for image,mode in sorted(current):
        new,old=current[image,mode],previous[image,mode]
        delta={field:int(new[field])-int(old[field]) for field in fields}
        for k,v in delta.items():
            counts[mode]['delta_totals'][k]+=v
        if any(delta.values()):
            counts[mode]['mismatch_images']+=1
            diffs.append({'image':image,'sequence':seq_of(image),'method':mode,
                          **{f'ref_{k}':int(old[k]) for k in fields},
                          **{f'current_{k}':int(new[k]) for k in fields},
                          **{f'delta_{k}':int(delta[k]) for k in fields}})
    return {'passed':not diffs,'counts':counts,'different_images':diffs,
            'reference_rows':len(previous),'current_rows':len(current)}


def verify_reference(full:dict,reference:dict) -> None:
    """For full 200-image run, same detector/full-frame controls as Step10D."""
    if reference.get('protocol') != 'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST':
        raise ValueError('Unexpected Step10D reference protocol')
    ref=reference['results']
    if full['images']!=ref['images'] or full['sequences']!=ref['unique_sequences']:
        raise ValueError('Reference evaluation sample/count mismatch')
    for mode in ('full_bicubic_640','full_large_fsrcnn_640'):
        for key in ('small_gt','small_tp','all_tp','fp'):
            if int(full['methods'][mode][key]) != int(ref['methods'][mode][key]):
                raise ValueError(f'Frozen {mode} differs from Step10D reference on {key}; investigate')



def verify_reference_provenance(reference:dict, manifest_sha:str,
                                large_sr_sha:str, frozen_shas:dict[str,str]) -> None:
    """Step11F must evaluate the EXACT models / split used in Step10D."""
    if reference.get('protocol') != 'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST':
        raise ValueError('Unexpected Step10D reference protocol')
    if reference.get('split_manifest_sha256') != manifest_sha:
        raise ValueError('Frozen development manifest differs from Step10D')
    ckpts=reference.get('evaluated_artifacts',{})
    if ckpts.get('large_sr_checkpoint',{}).get('sha256') != large_sr_sha:
        raise ValueError('Large FSRCNN differs from Step10D reference')
    ref_frozen=reference.get('frozen_checkpoints',{})
    for key in ('predictor_checkpoint','detector_checkpoint','sr_checkpoint'):
        if ref_frozen.get(key,{}).get('sha256') != frozen_shas[key]:
            raise ValueError(f'Frozen checkpoint differs from Step10D: {key}')


def write_csv(path:Path,rows:list[dict]) -> None:
    if not rows or path.exists():
        raise ValueError('Cannot save empty CSV or overwrite an existing CSV')
    with path.open('w',encoding='utf-8',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args:argparse.Namespace) -> dict:
    # Lazy project imports permit unit testing pure math/selection without YOLO.
    from prepare_development_split import sequence_id, training_sequences,check_manifest,sha256
    from evaluate_selective_sr import choose_tiles
    from probe_roi_utility import roi_grid, prediction_records, detection_counts
    from evaluate_bicubic_baseline import ground_truth
    from benchmark_selective_fsrcnn import load_sr_checkpoint
    from utitherm.sr.bicubic_first_fsrcnn_step11a import render_bicubic_first_fsrcnn
    from utitherm.sr.roi_fsrcnn import render_fsrcnn

    if args.out.exists() or args.limit<0 or not 3<=args.repeats<=10:
        raise ValueError('Output directory must be new; limit>=0, repeats 3..10')
    exp_path=args.config
    exp=yaml.safe_load(exp_path.read_text(encoding='utf-8'))
    base_path=Path(exp['base_development_config'])
    base=yaml.safe_load(base_path.read_text(encoding='utf-8'))
    manifest_path=Path(exp['frozen_manifest'])
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    validate_protocol(exp,base,manifest)
    train=Path(base['train_lr']); val=Path(base['eval_lr']); labels=val.parent.parent/'labels'/'val'
    tr_seq=training_sequences(train,Path(base['utility_csv']),Path(base['split_manifest']))
    names=check_manifest(manifest,val,labels,tr_seq,Path(base['utility_csv']),Path(base['split_manifest']))
    if len(names)!=200 or len({sequence_id(i) for i in names})!=17:
        raise ValueError('Frozen manifest returned an unexpected sample')
    for key in ('predictor_checkpoint','detector_checkpoint','sr_checkpoint'):
        file=Path(base[key]); checksum=sha256(file)
        if checksum!=manifest['checkpoint_sha256'][key]:
            raise ValueError(f'Frozen checkpoint mismatch: {key}')
    sr_ckpt=Path(exp['large_sr_checkpoint']); reference_path=Path(exp['reference_step10d_summary'])
    if not sr_ckpt.is_file() or not reference_path.is_file():
        raise FileNotFoundError('Missing Large FSRCNN checkpoint or Step10D reference summary')
    ref=json.loads(reference_path.read_text(encoding='utf-8'))
    reference_csv=reference_path.with_name('detection_per_image.csv')
    if not reference_csv.is_file():
        raise FileNotFoundError(f'Step10D per-image CSV required for diagnostic: {reference_csv}')
    frozen_shas={key:sha256(Path(base[key]))
                 for key in ('predictor_checkpoint','detector_checkpoint','sr_checkpoint')}
    verify_reference_provenance(ref,sha256(manifest_path),sha256(sr_ckpt),frozen_shas)
    if args.limit:
        names=names[:args.limit]
    if len(names)<2 or len({sequence_id(x) for x in names})<2:
        raise ValueError('Need two frames from distinct video sequences')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for comparable detector/SR timing')
    cv2.setNumThreads(1)
    device=torch.device('cuda:0')
    sr=load_sr_checkpoint(sr_ckpt,device)
    from ultralytics import YOLO
    detector=YOLO(str(base['detector_checkpoint']))
    tiles=roi_grid(640,512,4,4)
    if len(tiles)!=16:
        raise ValueError('Incorrect ROI tiling')
    # Audit cheap contrast against exact Step11D feature extraction for the smoke.
    if args.limit and args.limit<=8:
        from audit_bicubic_first_predictability_step11d import extract_frame_features,FEATURES,choose_indices
        for n in names:
            gray=cv2.imread(str(val/n),cv2.IMREAD_GRAYSCALE)
            ours=contrast_scores(gray,tiles)
            previous=extract_frame_features(gray,tiles)[:,FEATURES.index('tile_std')]
            if not np.array_equal(ours,previous) or topk(ours,4)!=choose_indices(previous,4):
                raise RuntimeError(f'Contrast feature/ranking differs from Step11D: {n}')
    dummy=np.zeros(SHAPE,np.uint8)
    for _ in range(2):
        for mode in MODES:
            chosen=[0,5,10,15] if 'first_h10' in mode else []
            frame,_=render(mode,dummy,tiles,chosen,sr,device,
                           render_bicubic_first_fsrcnn,render_fsrcnn)
            prediction_records(detector,frame,base)
    synchronize(device)
    trials=[];dets=[]
    for idx,image in enumerate(names,1):
        gray=cv2.imread(str(val/image),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=SHAPE or gray.dtype!=np.uint8:
            raise ValueError(f'Invalid LR frame {image}')
        selections={mode:tuple(selected_tiles(mode,gray,image,tiles,int(base['seed']),choose_tiles))
                    for mode in MODES}
        first={}
        for rep in range(args.repeats):
            order=list(MODES)
            random.Random(int(base['seed'])+idx*1073+rep).shuffle(order)
            for mode in order:
                row,preds=timed_trial(mode,gray,image,tiles,int(base['seed']),sr,detector,base,device,
                                      choose_tiles,render_bicubic_first_fsrcnn,render_fsrcnn,
                                      prediction_records,selections)
                trials.append({'image':image,'sequence':sequence_id(image),'repeat':rep,
                               'method':mode,**{k:round(float(v),6) for k,v in row.items()}})
                if rep==0: first[mode]=preds
        gts=ground_truth(labels/(Path(image).stem+'.txt'))
        for mode in MODES:
            dets.append({'image':image,'sequence':sequence_id(image),'method':mode,
                         **detection_counts(gts,first[mode],.25)})
        if idx==1 or idx%10==0 or idx==len(names):
            print(f'Evaluated {idx}/{len(names)} frozen development frames',flush=True)
    stats=summarize(dets,trials,args.repeats,float(exp['false_positive_penalty']),
                    int(exp['bootstrap_seed']),int(exp['bootstrap_iterations']))
    control_audit=None
    failure=None
    if not args.limit:
        control_audit=reference_control_differences(dets,reference_csv,ref)
        try:
            verify_reference(stats,ref)
        except ValueError as exc:
            failure=str(exc)
        if not control_audit['passed'] and failure is None:
            failure='Per-image Step10D controls differ, even though aggregate totals match'
    report={'task':'Step11F: ACTUAL combined four-ROI Bicubic-first Large FSRCNN + GPU timing',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'protocol':'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST',
            'config_path':str(exp_path),'config_sha256':sha256(exp_path),
            'base_config_path':str(base_path),'base_config_sha256':sha256(base_path),
            'manifest_path':str(manifest_path),'manifest_sha256':sha256(manifest_path),
            'reference_step10d_path':str(reference_path),
            'reference_step10d_sha256':sha256(reference_path),
            'frozen_checkpoint_hashes_verified':True,
            'frozen_checkpoint_sha256':{key:sha256(Path(base[key]))
                 for key in ('predictor_checkpoint','detector_checkpoint','sr_checkpoint')},
            'large_sr_checkpoint':str(sr_ckpt),'large_sr_sha256':sha256(sr_ckpt),
            'reference_full_controls_verified':bool(not args.limit and failure is None),
            'reference_control_diagnostic':({k:v for k,v in control_audit.items() if k!='different_images'}
                                             if control_audit is not None else None),
            'reference_integrity_failure':failure,
            'results':stats}
    args.out.mkdir(parents=True)
    write_csv(args.out/'detection_per_image.csv',dets)
    write_csv(args.out/'timing_per_trial.csv',trials)
    if control_audit is not None and control_audit['different_images']:
        write_csv(args.out/'reference_control_differences.csv',control_audit['different_images'])
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('\nSTEP 11F — ACTUAL four-ROI results on REUSED FLIR development set (NOT independent):')
    for mode in MODES:
        row=stats['methods'][mode]
        print(f'  {mode:32s} smallTP={row["small_tp"]:4} FP={row["fp"]:4} '
              f'recall={row["small_recall_conf_025"]:.3f} time={row["mean_image_median_total_ms"]:.3f} ms')
    for k,v in stats['comparisons'].items():
        print(f'  {k}: ΔTP={v["delta_small_tp"]:+d} ΔFP={v["delta_fp"]:+d} '
              f'ΔU={v["delta_utility_total"]:+.2f} CI/image={v["sequence_bootstrap_utility_per_image_95ci"]}')
    print(f'SAVED — {args.out/"summary.json"}',flush=True)
    if failure:
        print('REFERENCE CHECK FAILED — diagnostics have been saved, but the run is NOT verified.',flush=True)
        for mode,data in control_audit['counts'].items():
            print(f'  {mode}: changed_images={data["mismatch_images"]}, '
                  f'delta_totals={data["delta_totals"]}',flush=True)
        print(f'  Inspect {args.out/"reference_control_differences.csv"}',flush=True)
        raise ValueError(f'{failure}; see saved Step11F reference diagnostics. '
                         'Do NOT use this run as a verified reference-matched result.')
    return report


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/bicubic_first_combined_step11f.yaml'))
    p.add_argument('--limit',type=int,default=8,help='0=all 200 frozen development images')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':
    main()
