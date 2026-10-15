"""Step11I: verified deterministic Bicubic-vs-Contrast4 confidence audit.

Use all Step11G-reproduced post-NMS detections at base confidence 0.25;
re-evaluate *both* methods offline at FIXED thresholds >=0.25, preserving
class-aware matching and exposing retained GT identities. No inference at
higher detector confidence; this is a retrospective post-NMS cutoff audit.

NOT a clean/independent test, threshold search, calibration claim or mAP.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'src'))
import evaluate_bicubic_first_combined_step11f as step11f
import evaluate_bicubic_first_deterministic_step11g as step11g
import audit_spatial_false_positives_step11h as step11h

PROTOCOL = 'STEP11I_LOCKED_DETERMINISTIC_POSTNMS_CONFIDENCE_AUDIT_V1'
BASE = 'full_bicubic_640'
CONTRAST = 'contrast4_bicubic_first_h10'
METHODS = (BASE, CONTRAST)
THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.50)


def validate_thresholds(thresholds: list[float], min_conf: float = 0.25) -> tuple[float, ...]:
    values = tuple(float(t) for t in thresholds)
    if values != THRESHOLDS or not all(np.isfinite(t) and min_conf <= t <= 1 for t in values):
        raise ValueError('Step11I thresholds must be exactly fixed predeclared [0.25,0.30,0.35,0.40,0.50]')
    return values


def match_detail(gts: list[dict], preds: list[dict], threshold: float, box_iou):
    """Return matched GT→prediction, FP indices; identical greedy tie semantics to Step4.

    GT IoU threshold is 0.50, not `threshold` (the latter is confidence).
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid confidence')
    matched = {}
    fps = []
    order = sorted(range(len(preds)), key=lambda i: float(preds[i]['conf']), reverse=True)
    for pi in order:
        pred = preds[pi]
        if float(pred['conf']) < threshold:
            continue
        best, best_iou = None, 0.5
        for gi, gt in enumerate(gts):
            if gi in matched or int(gt['cls']) != int(pred['cls']):
                continue
            iou = float(box_iou(gt['box'], pred['box']))
            if iou >= best_iou:
                best, best_iou = gi, iou
        if best is None:
            fps.append(pi)
        else:
            matched[best] = pi
    small = {gi for gi in matched if gts[gi]['size'] == 'small'}
    return {'matched': matched, 'fp_indices': fps, 'small_gt_indices': small,
            'small_tp': len(small), 'all_tp': len(matched), 'fp': len(fps),
            'small_gt': sum(gt['size']=='small' for gt in gts)}


def compare_against_official(detail:dict, gts:list[dict], preds:list[dict], threshold:float,
                             official_count_fn, official_match_fn):
    ref=official_count_fn(gts, preds, threshold)
    for key in step11g.METRICS:
        if int(detail[key]) != int(ref[key]):
            raise RuntimeError(f'Matcher mismatch for {key} at confidence {threshold}: {detail[key]} vs {ref[key]}')
    expected_matched, expected_fp=official_match_fn(gts,preds,threshold)
    if set(detail['matched'])!=expected_matched or len(detail['fp_indices'])!=expected_fp:
        raise RuntimeError('GT identities differ from official Step4 class-aware matching')


def delta_records(base:dict, contrast:dict, fp_penalty:float) -> dict:
    if base['small_gt']!=contrast['small_gt']:
        raise ValueError('Different GT counts')
    gained=contrast['small_gt_indices']-base['small_gt_indices']
    lost=base['small_gt_indices']-contrast['small_gt_indices']
    if len(gained)-len(lost)!=contrast['small_tp']-base['small_tp']:
        raise AssertionError('TP identity conservation failure')
    dfp=contrast['fp']-base['fp']
    return {'gained_small_gt':len(gained), 'lost_small_gt':len(lost),
            'delta_small_tp':len(gained)-len(lost), 'delta_fp':dfp,
            'delta_utility':round((len(gained)-len(lost))-fp_penalty*dfp,8)}


def validate_base_row(row:dict, reference:dict, identity:tuple, threshold:float):
    if threshold!=THRESHOLDS[0]:
        return
    for key in step11g.METRICS:
        if int(row[key]) != int(reference[identity][key]):
            raise RuntimeError(f'Base threshold disagreeing with verified Step11G {identity} field {key}')


def summarize(per_image:list[dict], thresholds:tuple[float,...], fp_penalty:float,
              bootstrap_seed:int, bootstrap_iterations:int) -> dict:
    by = defaultdict(dict)
    sequences = defaultdict(list)
    for r in per_image:
        ident=(float(r['threshold']),r['image'],r['method'])
        if ident in by:
            raise ValueError('Duplicate image/method/threshold')
        by[ident]=r
        if r['image'] not in sequences[r['sequence']]:
            sequences[r['sequence']].append(r['image'])
    all_images={im for ims in sequences.values() for im in ims}
    if len(all_images)!=sum(len(i) for i in sequences.values()):
        raise ValueError('Image belongs to multiple sequences')
    if len(sequences)<2:
        raise ValueError('Cluster bootstrap requires two sequences')
    output={}
    for n,t in enumerate(thresholds):
        row_methods={}
        for method in METHODS:
            subset=[by.get((t,im,method)) for im in sorted(all_images)]
            if any(s is None for s in subset):
                raise ValueError(f'Missing {method} records at confidence {t}')
            totals={k:sum(int(s[k]) for s in subset) for k in ('small_gt','small_tp','all_tp','fp')}
            totals['small_recall']=round(totals['small_tp']/totals['small_gt'],6) if totals['small_gt'] else None
            totals['absolute_utility_proxy']=round(totals['small_tp']-fp_penalty*totals['fp'],6)
            row_methods[method]=totals
        diff={im:delta_records(
            {'small_gt':int(by[t,im,BASE]['small_gt']),
             'small_gt_indices':set(json.loads(by[t,im,BASE]['matched_small_gt_indices'])),
             'small_tp':int(by[t,im,BASE]['small_tp']),'fp':int(by[t,im,BASE]['fp'])},
            {'small_gt':int(by[t,im,CONTRAST]['small_gt']),
             'small_gt_indices':set(json.loads(by[t,im,CONTRAST]['matched_small_gt_indices'])),
             'small_tp':int(by[t,im,CONTRAST]['small_tp']),'fp':int(by[t,im,CONTRAST]['fp'])},
            fp_penalty) for im in all_images}
        if sum(v['delta_fp'] for v in diff.values())!=row_methods[CONTRAST]['fp']-row_methods[BASE]['fp']:
            raise AssertionError('FP conservation')
        vals={im:float(diff[im]['delta_utility']) for im in all_images}
        ci=step11f.cluster_ci(vals,sequences,bootstrap_seed+n*37,bootstrap_iterations)
        output[f'{t:.2f}']={
            'methods':row_methods,
            'contrast_vs_bicubic':{
                'gained_small_gt':sum(v['gained_small_gt'] for v in diff.values()),
                'lost_small_gt':sum(v['lost_small_gt'] for v in diff.values()),
                'delta_small_tp':row_methods[CONTRAST]['small_tp']-row_methods[BASE]['small_tp'],
                'delta_fp':row_methods[CONTRAST]['fp']-row_methods[BASE]['fp'],
                'delta_utility_total':round(sum(vals.values()),6),
                'delta_utility_per_image':round(sum(vals.values())/len(vals),6),
                'sequence_bootstrap_utility_per_image_95ci_exploratory':ci,
                'image_wins_ties_losses':{
                    'wins':sum(v>0 for v in vals.values()),
                    'ties':sum(v==0 for v in vals.values()),
                    'losses':sum(v<0 for v in vals.values()),
                },
            },
        }
    return output


def write_csv(path:Path, rows:list[dict], fields:list[str]):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='raise')
        writer.writeheader();writer.writerows(rows)


def run(args):
    if args.out.exists() or args.limit<0 or args.limit>200:
        raise ValueError('Output must be a NEW directory; --limit must be in 0..200')
    flags=step11g.setup_determinism()
    from prepare_development_split import sequence_id,training_sequences,check_manifest,sha256
    from evaluate_selective_sr import choose_tiles
    from probe_roi_utility import roi_grid,prediction_records,detection_counts
    from evaluate_bicubic_baseline import ground_truth,match_predictions,box_iou
    from benchmark_selective_fsrcnn import load_sr_checkpoint
    from utitherm.sr.bicubic_first_fsrcnn_step11a import render_bicubic_first_fsrcnn
    from utitherm.sr.roi_fsrcnn import render_fsrcnn

    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not isinstance(cfg,dict):
        raise ValueError('Invalid Step11I config')
    thresholds=validate_thresholds(cfg['thresholds'])
    if cfg['base_confidence']!=0.25 or cfg['fp_penalty']!=0.25:
        raise ValueError('Frozen confidence or FP penalty changed')
    if cfg['expected_images']!=200 or cfg['expected_sequences']!=17:
        raise ValueError('Frozen dataset protocol changed')
    n_boot=int(cfg['bootstrap_iterations']);seed=int(cfg['bootstrap_seed'])
    if n_boot<1000 or n_boot>10000:
        raise ValueError('Need 1000..10000 bootstrap iterations')
    cfg_g=Path(cfg['step11g_config'])
    g_data=yaml.safe_load(cfg_g.read_text(encoding='utf-8'))
    base_file=Path(g_data['base_development_config'])
    base=yaml.safe_load(base_file.read_text(encoding='utf-8'))
    manifest_file=Path(g_data['frozen_manifest'])
    manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
    step11f.validate_protocol(g_data,base,manifest)
    a=Path(cfg['reference_step11g_summary']);b=Path(cfg['verified_step11g_summary'])
    prior=json.loads(a.read_text(encoding='utf-8')); ref=json.loads(b.read_text(encoding='utf-8'))
    # Verifies source A/B hashes, counts, protocol, version, config and entire 4-mode CSV equality.
    reference_counts,reference_fps=step11h.validate_reference(
        ref,prior,cfg_g,g_data,
        {'expected_images':200,'expected_sequences':17,'confidence':0.25,'fp_penalty':0.25},a,b)
    counts=step11h.record_map(reference_counts)
    fingerprints=step11h.record_map(reference_fps)
    sr_path=Path(g_data['large_sr_checkpoint'])
    if sha256(sr_path)!=ref['large_sr_sha256'] or sha256(Path(base['detector_checkpoint']))!=ref['detector_sha256']:
        raise RuntimeError('Frozen checkpoint hash changed')
    if sha256(base_file)!=ref['base_config_sha256'] or sha256(manifest_file)!=ref['manifest_sha256']:
        raise RuntimeError('Frozen config or manifest SHA changed')
    val=Path(base['eval_lr']);train=Path(base['train_lr']);labels=val.parent.parent/'labels'/'val'
    trseq=training_sequences(train,Path(base['utility_csv']),Path(base['split_manifest']))
    names=check_manifest(manifest,val,labels,trseq,Path(base['utility_csv']),Path(base['split_manifest']))
    if len(names)!=200 or len({sequence_id(i) for i in names})!=17:
        raise RuntimeError('Frozen 200/17 dataset mismatch')
    if args.limit:
        names=names[:args.limit]
    if len(names)<2 or len({sequence_id(i) for i in names})<2:
        raise ValueError('At least two frames and sequences required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU needed')
    device=torch.device('cuda:0')
    model=load_sr_checkpoint(sr_path,device)
    from ultralytics import YOLO
    detector=YOLO(str(base['detector_checkpoint']))
    env=step11g.env_snapshot()
    if env!=ref['environment']:
        raise RuntimeError('Runtime differs from Step11G verified reference')
    tiles=roi_grid(640,512,4,4)
    dummy=np.zeros(step11f.SHAPE,np.uint8)
    for _ in range(2):
        for method in step11f.MODES:
            selected=[0,5,10,15] if '_first_h10' in method else []
            frame,_=step11f.render(method,dummy,tiles,selected,model,device,
                                   render_bicubic_first_fsrcnn,render_fsrcnn)
            prediction_records(detector,frame,base)
    step11f.synchronize(device)

    per_image=[];fingerprint_mismatches=[];tp_scores=[];fp_scores=[]
    class_counts=defaultdict(Counter)
    for n,name in enumerate(names,1):
        gray=cv2.imread(str(val/name),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=(256,320) or gray.dtype!=np.uint8:
            raise ValueError(f'Bad LR frame: {name}')
        gts=ground_truth(labels/(Path(name).stem+'.txt'))
        order=list(step11f.MODES)
        random.Random(int(base['seed'])+n*1073).shuffle(order)
        predictions={}
        for method in order:
            chosen=step11f.selected_tiles(method,gray,name,tiles,int(base['seed']),choose_tiles)
            frame,_=step11f.render(method,gray,tiles,chosen,model,device,
                                   render_bicubic_first_fsrcnn,render_fsrcnn)
            preds=prediction_records(detector,frame,base)
            identity=(name,method)
            expected=fingerprints[identity]
            observed={
                'rendered_sha256':step11g.image_fingerprint(frame),
                'predictions_sha256':step11g.prediction_fingerprint(preds),
                'num_predictions':str(len(preds)),
                'selected_tiles':json.dumps(chosen,separators=(',',':')),
            }
            for field,value in observed.items():
                if str(value)!=str(expected[field]):
                    fingerprint_mismatches.append({'image':name,'method':method,
                                                   'field':field,'expected':expected[field],'actual':value})
            if method in METHODS:
                predictions[method]=preds
        for method in METHODS:
            preds=predictions[method]
            for conf in thresholds:
                detailed=match_detail(gts,preds,conf,box_iou)
                compare_against_official(detailed,gts,preds,conf,detection_counts,match_predictions)
                identity=(name,method)
                validate_base_row(detailed,counts,identity,conf)
                per_image.append({'image':name,'sequence':sequence_id(name),'method':method,
                    'threshold':f'{conf:.2f}','small_gt':detailed['small_gt'],
                    'small_tp':detailed['small_tp'],'all_tp':detailed['all_tp'],
                    'fp':detailed['fp'],'matched_small_gt_indices':json.dumps(sorted(detailed['small_gt_indices']),separators=(',',':'))})
                # Aggregate small-object detection and FP by class at each fixed threshold.
                class_key=(f'{conf:.2f}',method)
                for obj in gts:
                    if obj['size']=='small':
                        class_counts[class_key]['small_gt:'+str(obj['cls'])]+=1
                for gi in detailed['small_gt_indices']:
                    class_counts[class_key]['small_tp:'+str(gts[gi]['cls'])]+=1
                for pi in detailed['fp_indices']:
                    class_counts[class_key]['fp:'+str(preds[pi]['cls'])]+=1
                # Diagnostics for actual matcher TP/FP confidence (before threshold filtering).
                if conf==thresholds[0]:
                    for gi,pi in detailed['matched'].items():
                        if gts[gi]['size']=='small':
                            tp_scores.append({'image':name,'method':method,'class_id':gts[gi]['cls'],
                                'confidence':round(float(preds[pi]['conf']),8)})
                    for pi in detailed['fp_indices']:
                        fp_scores.append({'image':name,'method':method,'class_id':int(preds[pi]['cls']),
                            'confidence':round(float(preds[pi]['conf']),8)})
        if n==1 or n%20==0 or n==len(names):
            print(f'Step11I verified fixed-threshold audit {n}/{len(names)} images',flush=True)
    if fingerprint_mismatches:
        args.out.mkdir(parents=True)
        write_csv(args.out/'reference_mismatches.csv',fingerprint_mismatches,
                  ['image','method','field','expected','actual'])
        raise RuntimeError(f'{len(fingerprint_mismatches)} Step11G fingerprint mismatches; report saved, DO NOT interpret')
    results=summarize(per_image,thresholds,float(cfg['fp_penalty']),seed,n_boot)
    at_base=results[f'{thresholds[0]:.2f}']
    if args.limit==0:
        for method in METHODS:
            for key in step11g.METRICS:
                if int(at_base['methods'][method][key]) != int(ref['results']['methods'][method][key]):
                    raise RuntimeError(f'Aggregate conf .25 differs from Step11G for {method}/{key}')
        assert at_base['contrast_vs_bicubic']['delta_small_tp']==15
        assert at_base['contrast_vs_bicubic']['delta_fp']==88
        assert at_base['contrast_vs_bicubic']['delta_utility_total']==-7.0
    summary={'protocol':PROTOCOL,'created_utc':datetime.now(timezone.utc).isoformat(),
        'images':len(names),'sequences':len({sequence_id(x) for x in names}),
        'config_sha256':sha256(args.config),'step11g_reference_sha256':sha256(a),
        'step11g_verified_sha256':sha256(b),'environment':env,
        'deterministic_settings':flags,'fingerprints_verified':not fingerprint_mismatches,
        'fingerprint_mismatches':len(fingerprint_mismatches),
        'inference_confidence':float(base['confidence']),'gt_match_iou':0.5,
        'confidence_thresholds':[float(t) for t in thresholds],
        'methods':list(METHODS),'results_by_confidence':results,
        'warnings':[
            'Previously analyzed FLIR validation DEVELOPMENT images: exploratory, NOT independent.',
            'All thresholds are POST-NMS filters on the same YOLO outputs at conf=0.25; no higher-confidence inference was rerun.',
            'GT matching is class-aware greedy 1:1 at IoU>=0.50. FP is relative to available labels; unlabeled real objects may be FPs.',
            'Comparison of FP candidates and GT matches is descriptive; no demonstrated causal intervention or calibrated probabilities.',
            'Applying the same threshold to both methods is not proof the contrast method is better at a different operating point.',
            'Threshold choice based on this repeatedly inspected development set will overfit; do not call any threshold a final policy.',
            'Higher confidence threshold does not reduce underlying GPU inference latency in this offline audit.',
        ]}
    args.out.mkdir(parents=True)
    fields=['image','sequence','method','threshold','small_gt','small_tp','all_tp','fp','matched_small_gt_indices']
    write_csv(args.out/'per_image_confidence.csv',per_image,fields)
    write_csv(args.out/'matched_small_tp_confidence_at_025.csv',tp_scores,['image','method','class_id','confidence'])
    write_csv(args.out/'false_positive_confidence_at_025.csv',fp_scores,['image','method','class_id','confidence'])
    per_class=[]
    class_names={int(k):str(v) for k,v in detector.names.items()}
    for conf in thresholds:
        for method in METHODS:
            vals=class_counts[(f'{conf:.2f}',method)]
            for cls in range(9):
                per_class.append({'confidence_threshold':f'{conf:.2f}','method':method,
                                  'class_id':cls,'class_name':class_names[cls],
                                  'small_gt':vals['small_gt:'+str(cls)],
                                  'small_tp':vals['small_tp:'+str(cls)],
                                  'fp':vals['fp:'+str(cls)]})
    write_csv(args.out/'per_class_confidence.csv',per_class,
              ['confidence_threshold','method','class_id','class_name','small_gt','small_tp','fp'])
    if fingerprint_mismatches:
        write_csv(args.out/'reference_mismatches.csv',fingerprint_mismatches,
                  ['image','method','field','expected','actual'])
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('\nSTEP11I — fixed post-NMS confidence utility sweep (exploratory):')
    for t in thresholds:
        res=results[f'{t:.2f}']; b=res['methods'][BASE];c=res['methods'][CONTRAST]
        d=res['contrast_vs_bicubic']
        print(f'  conf={t:.2f}: bicubic TP={b["small_tp"]} FP={b["fp"]} | '
              f'contrast TP={c["small_tp"]} FP={c["fp"]} | '
              f'ΔTP={d["delta_small_tp"]:+d} ΔFP={d["delta_fp"]:+d} ΔU={d["delta_utility_total"]:+.2f}')
    print('SAVED —',args.out/'summary.json')
    if fingerprint_mismatches:
        raise RuntimeError(f'{len(fingerprint_mismatches)} fingerprint mismatches: STOP, do not use results')
    print('VERIFIED: all four Step11G rendering and prediction fingerprints match; conf=0.25 counts match.')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/step11i_confidence_utility_audit.yaml'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0,help='0=all 200; positive N for smoke/debug only')
    run(p.parse_args())

if __name__=='__main__':
    main()
