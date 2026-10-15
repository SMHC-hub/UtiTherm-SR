"""Step11H: deterministic, reference-fingerprint-verified spatial FP audit.

Exploratory *reused* 200-image FLIR development set. No novel method or
independent testing claim. Audits Bicubic vs contrast4 only. A spatial association
is not proof that the neural edit caused a new FP (global YOLO/NMS interactions).
Run with Docker -e CUBLAS_WORKSPACE_CONFIG=:4096:8.
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
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import evaluate_bicubic_first_combined_step11f as step11f
import evaluate_bicubic_first_deterministic_step11g as step11g

PROTOCOL = 'STEP11H_VERIFIED_DETERMINISTIC_SPATIAL_FP_AUDIT_V1'
BASE = 'full_bicubic_640'
CONTRAST = 'contrast4_bicubic_first_h10'
AUDITED = (BASE, CONTRAST)


def match_fp_indices(gts: list[dict], preds: list[dict], confidence: float,
                     box_iou=None, gt_iou: float = 0.5) -> list[int]:
    """Mirror Step4 match_predictions exactly, retaining FP prediction indices.

    The official count uses sorted confidence, one-to-one class-aware GT matching.
    We verify its FP count outside this function. Stable ties use input order.
    """
    if box_iou is None:
        from evaluate_bicubic_baseline import box_iou
    if not 0 <= confidence <= 1 or not 0 < gt_iou <= 1:
        raise ValueError('Invalid confidence or matching IoU')
    matched = set()
    fps = []
    order = sorted(range(len(preds)), key=lambda i: float(preds[i]['conf']), reverse=True)
    for pi in order:
        pred = preds[pi]
        if pred['conf'] < confidence:
            continue
        best, best_iou = None, gt_iou
        for gi, gt in enumerate(gts):
            if gi in matched or gt['cls'] != pred['cls']:
                continue
            overlap = box_iou(gt['box'], pred['box'])
            if overlap >= best_iou:
                best, best_iou = gi, overlap
        if best is None:
            fps.append(pi)
        else:
            matched.add(best)
    return fps


def pair_false_positives(base_preds: list[dict], cont_preds: list[dict],
                         base_fp: list[int], cont_fp: list[int], iou: float,
                         box_iou=None) -> tuple[list[tuple[int,int,float]], list[int], list[int]]:
    """Greedy one-to-one class-aware spatial pairing of already-classified FPs.

    Unpaired contrast FPs are 'new candidates'; unpaired baseline FPs 'removed'.
    A shifted detection may be unmatched. Threshold is *not* label-matching IoU.
    """
    if box_iou is None:
        from evaluate_bicubic_baseline import box_iou
    if not 0 < iou <= 1:
        raise ValueError('FP pairing IoU must be in (0,1]')
    if len(set(base_fp)) != len(base_fp) or len(set(cont_fp)) != len(cont_fp):
        raise ValueError('Duplicate FP index')
    if any(not 0 <= x < len(base_preds) for x in base_fp) or any(not 0 <= x < len(cont_preds) for x in cont_fp):
        raise ValueError('Invalid FP index')
    candidates=[]
    for bi in base_fp:
        for ci in cont_fp:
            if int(base_preds[bi]['cls']) != int(cont_preds[ci]['cls']):
                continue
            score=float(box_iou(base_preds[bi]['box'], cont_preds[ci]['box']))
            if score>=iou:
                candidates.append((-score,bi,ci))
    pairs=[]; used_base=set(); used_cont=set()
    for neg_score,bi,ci in sorted(candidates):
        if bi not in used_base and ci not in used_cont:
            pairs.append((bi,ci,-neg_score)); used_base.add(bi); used_cont.add(ci)
    added=sorted(set(cont_fp)-used_cont)
    removed=sorted(set(base_fp)-used_base)
    if len(added)-len(removed)!=len(cont_fp)-len(base_fp):
        raise AssertionError('FP conservation violated')
    return pairs,added,removed


def locate_fp(box_lr, selected: list[int], tiles_hr: list[tuple[int,int,int,int]],
              boundary_hr: int) -> dict:
    """Location of LR-space FP relative to HR neural-edit tile footprints.

    Uses centre for exclusive mutually exclusive categories. Also reports a
    separate bbox-intersection fraction (non-exclusive association).
    Halo operates on CNN context; actual output edit is within the tile only.
    """
    if len(box_lr)!=4 or not all(np.isfinite(x) for x in box_lr):
        raise ValueError('Invalid FP box')
    x0,y0,x1,y1=(float(v)*2 for v in box_lr)
    if not (x0<x1 and y0<y1 and 0<=x0<x1<=640 and 0<=y0<y1<=512):
        raise ValueError('FP box outside original LR image limits')
    if not selected or len(set(selected))!=len(selected) or not 0<=boundary_hr<=32:
        raise ValueError('Invalid chosen tiles or boundary')
    if any(not 0<=i<len(tiles_hr) for i in selected):
        raise ValueError('Unknown tile index')
    cx,cy=(x0+x1)/2,(y0+y1)/2
    area=(x1-x0)*(y1-y0)
    overlap=0.0
    where='outside_selected_tiles'
    owner=None
    for i in selected:
        tx0,ty0,tx1,ty1=tiles_hr[i]
        dx=max(0.0,min(x1,tx1)-max(x0,tx0))
        dy=max(0.0,min(y1,ty1)-max(y0,ty0))
        overlap+=dx*dy
        if tx0<=cx<tx1 and ty0<=cy<ty1:
            owner=i
            margin=min(cx-tx0,tx1-cx,cy-ty0,ty1-cy)
            where='selected_boundary_band' if margin<boundary_hr else 'selected_interior'
    if overlap>area+1e-5:
        raise AssertionError('Overlapping edit tiles?')
    return {'centroid_zone':where,'center_tile':owner,
            'box_selected_overlap_fraction':round(min(1.0,overlap/area),6),
            'bbox_overlaps_selected':int(overlap>0),
            'center_x_hr':round(cx,4),'center_y_hr':round(cy,4)}


def validate_reference(ref:dict, prior:dict, g_cfg:Path, g_data:dict,
                       cfg:dict, source_a:Path, source_b:Path):
    from prepare_development_split import sha256
    if ref.get('protocol')!=step11g.PROTOCOL or prior.get('protocol')!=step11g.PROTOCOL:
        raise ValueError('Unexpected Step11G reference protocol')
    if not ref.get('repeats_verified') or not ref.get('cross_container_verified') or ref.get('cross_run_mismatch_count')!=0:
        raise ValueError('Step11G B must be cross-container verified')
    if not prior.get('repeats_verified') or prior.get('cross_container_verified'):
        raise ValueError('Step11G A must be internally verified original capture')
    if int(ref.get('images',-1))!=int(cfg['expected_images']) or int(ref.get('sequences',-1))!=int(cfg['expected_sequences']):
        raise ValueError('Unexpected reference image or sequence count')
    if ref.get('config_sha256')!=sha256(g_cfg) or prior.get('config_sha256')!=sha256(g_cfg):
        raise ValueError('Step11G config SHA changed')
    if float(g_data['confidence']) != float(cfg['confidence']) or float(g_data['false_positive_penalty'])!=float(cfg['fp_penalty']):
        raise ValueError('Step11H thresholds not equivalent to Step11G')
    for key in ('config_sha256','base_config_sha256','manifest_sha256','large_sr_sha256',
                'detector_sha256','environment','methods','deterministic_settings'):
        if ref.get(key)!=prior.get(key):
            raise ValueError(f'Step11G reference pair provenance mismatch: {key}')
    det_a=step11g.csv_rows(source_a.parent/'detection_per_image.csv')
    det_b=step11g.csv_rows(source_b.parent/'detection_per_image.csv')
    fp_a=step11g.csv_rows(source_a.parent/'fingerprints_per_image.csv')
    fp_b=step11g.csv_rows(source_b.parent/'fingerprints_per_image.csv')
    if step11g.compare_rows(det_b,det_a,step11g.METRICS) or step11g.compare_rows(
        fp_b,fp_a,('selected_tiles','rendered_sha256','predictions_sha256','num_predictions')):
        raise ValueError('Step11G A/B report CSVs do not match; cannot attribute')
    expected=int(cfg['expected_images'])*len(step11f.MODES)
    if any(len(x)!=expected for x in (det_a,det_b,fp_a,fp_b)):
        raise ValueError('Step11G reference CSV records incomplete')
    return det_b,fp_b


def record_map(rows:list[dict], key:str='method') -> dict[tuple[str,str],dict]:
    mapping={}
    for row in rows:
        ident=(row['image'],row[key])
        if ident in mapping:
            raise ValueError(f'Duplicate reference row: {ident}')
        mapping[ident]=row
    return mapping


def write_csv(path:Path, rows:list[dict], fields:list[str]):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if args.out.exists() or args.limit not in (0,) and not 1 <= args.limit<=200:
        raise ValueError('Output must be a new directory and --limit 0..200')
    settings=step11g.setup_determinism()  # fail closed *before* CUDA init
    from prepare_development_split import sequence_id,training_sequences,check_manifest,sha256
    from evaluate_selective_sr import choose_tiles
    from probe_roi_utility import roi_grid,prediction_records,detection_counts
    from evaluate_bicubic_baseline import ground_truth
    from benchmark_selective_fsrcnn import load_sr_checkpoint
    from utitherm.sr.bicubic_first_fsrcnn_step11a import render_bicubic_first_fsrcnn
    from utitherm.sr.roi_fsrcnn import render_fsrcnn

    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not isinstance(cfg,dict):
        raise ValueError('Malformed Step11H config')
    config_step11g=Path(cfg['step11g_config'])
    g_cfg=yaml.safe_load(config_step11g.read_text(encoding='utf-8'))
    base_path=Path(g_cfg['base_development_config'])
    base=yaml.safe_load(base_path.read_text(encoding='utf-8'))
    frozen_path=Path(g_cfg['frozen_manifest'])
    manifest=json.loads(frozen_path.read_text(encoding='utf-8'))
    step11f.validate_protocol(g_cfg,base,manifest)
    a=Path(cfg['reference_step11g_summary']);b=Path(cfg['verified_step11g_summary'])
    prior=json.loads(a.read_text(encoding='utf-8'));ref=json.loads(b.read_text(encoding='utf-8'))
    all_counts,all_fps=validate_reference(ref,prior,config_step11g,g_cfg,cfg,a,b)
    counts=record_map(all_counts);fingerprints=record_map(all_fps)
    sr_path=Path(g_cfg['large_sr_checkpoint'])
    if sha256(sr_path)!=ref['large_sr_sha256'] or sha256(Path(base['detector_checkpoint']))!=ref['detector_sha256']:
        raise ValueError('Checkpoint hash changed since Step11G')
    if sha256(base_path)!=ref['base_config_sha256'] or sha256(frozen_path)!=ref['manifest_sha256']:
        raise ValueError('Frozen config/manifest hash changed')
    val=Path(base['eval_lr']);train=Path(base['train_lr']);labels=val.parent.parent/'labels'/'val'
    trseq=training_sequences(train,Path(base['utility_csv']),Path(base['split_manifest']))
    names=check_manifest(manifest,val,labels,trseq,Path(base['utility_csv']),Path(base['split_manifest']))
    if len(names)!=200 or len({sequence_id(x) for x in names})!=17:
        raise ValueError('Frozen manifest is not the full 200/17 set')
    if args.limit:
        names=names[:args.limit]
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required')
    device=torch.device('cuda:0')
    model=load_sr_checkpoint(sr_path,device)
    from ultralytics import YOLO
    detector=YOLO(str(base['detector_checkpoint']))
    env=step11g.env_snapshot()
    if env!=ref['environment']:
        raise ValueError('Runtime versions/device differ from Step11G reference')
    tiles=roi_grid(640,512,4,4)
    # Mirror Step11G fixed warmup, which can affect the first frame's kernels.
    dummy=np.zeros(step11f.SHAPE,np.uint8)
    for _ in range(2):
        for mode in step11f.MODES:
            picked=[0,5,10,15] if '_first_h10' in mode else []
            frame,_=step11f.render(mode,dummy,tiles,picked,model,device,
                                  render_bicubic_first_fsrcnn,render_fsrcnn)
            prediction_records(detector,frame,base)
    step11f.synchronize(device)
    per_img=[]; fp_rows=[]; mismatch=[]
    for n,name in enumerate(names,1):
        gray=cv2.imread(str(val/name),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=(256,320) or gray.dtype!=np.uint8:
            raise ValueError(f'Bad LR image {name}')
        gts=ground_truth(labels/(Path(name).stem+'.txt'))
        pick=step11f.selected_tiles(CONTRAST,gray,name,tiles,int(base['seed']),choose_tiles)
        results={}
        # Reproduce the exact Step11G repeat-0 MODE ordering and warmup.
        # Compute only first-repeat results, but execute the other two methods
        # in sequence so detector CUDA call history also matches the reference.
        order=list(step11f.MODES)
        random.Random(int(base['seed'])+n*1073).shuffle(order)
        for mode in order:
            chosen=step11f.selected_tiles(mode,gray,name,tiles,int(base['seed']),choose_tiles)
            frame,_=step11f.render(mode,gray,tiles,chosen,model,device,
                                  render_bicubic_first_fsrcnn,render_fsrcnn)
            preds=prediction_records(detector,frame,base)
            if mode not in AUDITED:
                continue
            key=(name,mode);expected=fingerprints[key]
            observed={'rendered_sha256':step11g.image_fingerprint(frame),
                      'predictions_sha256':step11g.prediction_fingerprint(preds),
                      'num_predictions':str(len(preds)),
                      'selected_tiles':json.dumps(chosen,separators=(',',':'))}
            for field in observed:
                if str(expected[field])!=str(observed[field]):
                    mismatch.append({'image':name,'method':mode,'field':field,
                                     'expected':str(expected[field]),'actual':str(observed[field])})
            rec=detection_counts(gts,preds,float(base['confidence']))
            for field in step11g.METRICS:
                if int(rec[field])!=int(counts[key][field]):
                    mismatch.append({'image':name,'method':mode,'field':field,
                                     'expected':str(counts[key][field]),'actual':str(rec[field])})
            results[mode]=(preds,match_fp_indices(gts,preds,float(base['confidence'])))
        bp,bfp=results[BASE];cp,cfp=results[CONTRAST]
        paired,added,removed=pair_false_positives(bp,cp,bfp,cfp,float(cfg['fp_pair_iou']))
        delta_tp=int(counts[(name,CONTRAST)]['small_tp'])-int(counts[(name,BASE)]['small_tp'])
        delta_fp=len(cfp)-len(bfp)
        if len(added)-len(removed)!=delta_fp:
            raise AssertionError('Conservation of net false positives failed')
        zones=Counter()
        for ci in added:
            p=cp[ci]
            location=locate_fp(p['box'],pick,tiles,int(cfg['boundary_hr_pixels']))
            zones[location['centroid_zone']]+=1
            fp_rows.append({'image':name,'sequence':sequence_id(name),'prediction_index':ci,
                            'cls':int(p['cls']),'confidence':round(float(p['conf']),8),
                            'box_lr_xyxy':json.dumps([round(float(v),5) for v in p['box']]),
                            'selected_tiles':json.dumps(pick),'paired_iou_threshold':cfg['fp_pair_iou'],
                            **location})
        per_img.append({'image':name,'sequence':sequence_id(name),
                        'baseline_fp':len(bfp),'contrast_fp':len(cfp),
                        'delta_fp':delta_fp,'delta_small_tp':delta_tp,
                        'delta_utility':round(delta_tp-float(cfg['fp_penalty'])*delta_fp,6),
                        'paired_fp':len(paired),'added_fp_candidates':len(added),
                        'removed_fp_candidates':len(removed),
                        'selected_tiles':json.dumps(pick),
                        'added_centroids_inside':zones['selected_interior'],
                        'added_centroids_boundary':zones['selected_boundary_band'],
                        'added_centroids_outside':zones['outside_selected_tiles']})
        if n==1 or n%20==0 or n==len(names):
            print(f'Step11H audited {n}/{len(names)} development frames',flush=True)
    if args.limit==0:
        assert sum(r['delta_fp'] for r in per_img)==88
        assert sum(r['delta_small_tp'] for r in per_img)==15
    by_seq=defaultdict(list)
    for r in per_img: by_seq[r['sequence']].append(r)
    sequence_rows=[]
    for seq,items in sorted(by_seq.items()):
        sequence_rows.append({'sequence':seq,'images':len(items),
                              'net_delta_fp':sum(x['delta_fp'] for x in items),
                              'added_fp_candidates':sum(x['added_fp_candidates'] for x in items),
                              'removed_fp_candidates':sum(x['removed_fp_candidates'] for x in items),
                              'net_delta_small_tp':sum(x['delta_small_tp'] for x in items)})
    summary={'protocol':PROTOCOL,'created_utc':datetime.now(timezone.utc).isoformat(),
             'deterministic_settings':settings,'step11g_verified_summary':str(b),
             'step11g_reference_summary':str(a),'step11g_verified_summary_sha256':sha256(b),
             'step11g_reference_summary_sha256':sha256(a),
             'config_sha256':sha256(args.config), 'images':len(names), 'sequences':len(by_seq),
             'reference_fingerprint_match':not mismatch, 'mismatch_count':len(mismatch),
             'results':{'added_fp_candidates':len(fp_rows),
                        'removed_fp_candidates':sum(x['removed_fp_candidates'] for x in per_img),
                        'net_delta_fp':sum(x['delta_fp'] for x in per_img),
                        'net_delta_small_tp':sum(x['delta_small_tp'] for x in per_img),
                        'net_utility':round(sum(x['delta_utility'] for x in per_img),6),
                        'added_fp_centroid_zones':dict(Counter(r['centroid_zone'] for r in fp_rows)),
                        'added_fp_bbox_intersects_any_selected':sum(int(r['bbox_overlaps_selected']) for r in fp_rows),
                        'images_with_positive_delta_fp':sum(r['delta_fp']>0 for r in per_img),
                        'images_with_negative_delta_fp':sum(r['delta_fp']<0 for r in per_img),
                        'images_with_zero_delta_fp':sum(r['delta_fp']==0 for r in per_img)},
             'limitations':['Retrospective FLIR validation development set, NOT independent.',
                'New/removed FP candidates matched by same-class IoU, not ground-truth causal identity.',
                'Spatial centroid association is NOT causal proof; NMS is global.',
                'Unlabeled real objects may be counted as FPs under available labels.',
                'Full Bicubic vs selected contrast4 only; not an intervention isolating feather effects.',
                'No retraining, predictor selection, or new performance claim.']}
    args.out.mkdir(parents=True)
    write_csv(args.out/'fp_candidates.csv',fp_rows,
              ['image','sequence','prediction_index','cls','confidence','box_lr_xyxy',
               'selected_tiles','paired_iou_threshold','centroid_zone','center_tile',
               'box_selected_overlap_fraction','bbox_overlaps_selected','center_x_hr','center_y_hr'])
    write_csv(args.out/'per_image_spatial_fp.csv',per_img,
              ['image','sequence','baseline_fp','contrast_fp','delta_fp','delta_small_tp',
               'delta_utility','paired_fp','added_fp_candidates','removed_fp_candidates',
               'selected_tiles','added_centroids_inside','added_centroids_boundary','added_centroids_outside'])
    write_csv(args.out/'per_sequence_spatial_fp.csv',sequence_rows,
              ['sequence','images','net_delta_fp','added_fp_candidates','removed_fp_candidates','net_delta_small_tp'])
    if mismatch:
        write_csv(args.out/'reference_mismatches.csv',mismatch,
                  ['image','method','field','expected','actual'])
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('STEP11H — retrospective spatial FP candidates, NOT causal attribution:')
    print('  net_delta_fp:',summary['results']['net_delta_fp'],
          ' added:',summary['results']['added_fp_candidates'],
          ' removed:',summary['results']['removed_fp_candidates'])
    print('  centroid locations:',summary['results']['added_fp_centroid_zones'])
    print('  verified fingerprint match:',not mismatch,' mismatches:',len(mismatch))
    print('SAVED —',args.out/'summary.json')
    if mismatch:
        raise RuntimeError('Step11H predictions differ from Step11G verified reference; DO NOT interpret spatial results')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/step11h_fp_spatial_audit.yaml'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0,help='0=all 200; for debugging only, 1..200')
    run(p.parse_args())

if __name__=='__main__':
    main()
