"""Step 11L: predeclared sequence-disjoint FLIR TEST evaluation (deterministic).

NO model tuning, NO test-based confidence selection. Evaluates all four frozen
Step11G modes, then post-NMS filters predictions at 0.25 and 0.40.
Confidence 0.25 vs 0.25 is the PRIMARY comparison; 0.40 studies are SECONDARY.

Requires: docker compose run --rm -e CUBLAS_WORKSPACE_CONFIG=:4096:8 research ...
Smoke: python scripts/evaluate_frozen_flir_test_step11l.py --limit 8 --out reports/test/step11l_smoke_a
Full:  python scripts/evaluate_frozen_flir_test_step11l.py --limit 0 --out reports/test/step11l_full_a
Verify: same command with a NEW --out and --compare prior/summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_bicubic_first_combined_step11f as step11f
import evaluate_bicubic_first_deterministic_step11g as step11g
import prepare_frozen_flir_test_step11k as step11k

PROTOCOL = 'STEP11L_PREDECLARED_SEQUENCE_DISJOINT_FLIR_VIDEO_TEST_V1'
METHODS = step11f.MODES
CONFIDENCES = (0.25, 0.40)
PRIMARY = ('contrast4_bicubic_first_h10',0.25,'full_bicubic_640',0.25)
SECONDARY = (
    ('contrast4_bicubic_first_h10',0.40,'full_bicubic_640',0.40),
    ('contrast4_bicubic_first_h10',0.40,'full_bicubic_640',0.25),
)
KEYS=('small_gt','small_tp','all_tp','fp')
FP_FIELDS=('selected_tiles','rendered_sha256','predictions_sha256','num_predictions')


def validate_frozen_config(c:dict,base:dict,gconfig:dict,kconfig:dict):
    if not isinstance(c,dict):
        raise ValueError('Expected Step11L mapping')
    exact={
        'expected_test_images':3749,'expected_test_sequences':8,'expected_test_boxes':62015,
        'budget_tiles':4,'halo':10,'feather_hr_px':8,'reference_prediction_confidence':0.25,
        'nms_iou':0.70,'matching_iou':0.50,'fp_penalty':0.25,'random_seed':42,
        'bootstrap_seed':143,'bootstrap_iterations':4000,
        'expected_untouched_test_manifest_sha256':'af97d628d3984eca440668603d775f3153bfd79962b4e8629bc1f6fed0b8a35c',
        'expected_step11k_config_sha256':'3303352fbc8036cc0dc8bafcb5f43477e96b0d274374a1eabeb989bc0e6318dc',
        'expected_source_coco_sha256':'8da913c6cf57fd148afdd2ec81716e6bc44f1869308b92d6b6280263788d487a',
        'primary_comparison':'contrast4_at_0.25_vs_bicubic_at_0.25',
    }
    for k,v in exact.items():
        if c.get(k)!=v:
            raise ValueError(f'Frozen Step11L protocol changed: {k}')
    if list(c.get('methods',[]))!=list(METHODS):
        raise ValueError('Four-arm method list/order changed')
    if tuple(map(float,c.get('confidence_thresholds',())))!=CONFIDENCES:
        raise ValueError('Confidence thresholds changed')
    if c.get('secondary_comparisons')!=[
        'contrast4_at_0.40_vs_bicubic_at_0.40',
        'contrast4_at_0.40_vs_bicubic_at_0.25']:
        raise ValueError('Secondary comparisons changed')
    if c['random_seed']!=base['seed'] or base['confidence']!=0.25 or base['nms_iou']!=0.7:
        raise ValueError('Step11G base detector settings differ')
    if base['imgsz']!=640 or base['false_positive_penalty']!=0.25 or str(base['device'])!='0':
        raise ValueError('Step11G baseline changed')
    if list(gconfig.get('methods',[]))!=list(METHODS) or gconfig.get('halo')!=10 or gconfig.get('feather_hr_px')!=8:
        raise ValueError('Step11G method config changed')
    step11k.checked_config(kconfig)
    return True


def validate_sealed_test(c:dict,source_summary:dict,manifest:list[dict],full:bool):
    if source_summary.get('protocol')!=step11k.PROTOCOL or source_summary.get('images')!=3749:
        raise ValueError('Expected complete Step11K prepared dataset')
    if (source_summary.get('sequences')!=8 or source_summary.get('converted_yolo_boxes')!=62015
        or source_summary.get('manifest_content_sha256')!=c['expected_untouched_test_manifest_sha256']
        or source_summary.get('config_sha256')!=c['expected_step11k_config_sha256']
        or source_summary.get('source_coco_sha256')!=c['expected_source_coco_sha256']
        or source_summary.get('detector_inference_performed') is not False):
        raise ValueError('Prepared test provenance is not the sealed Step11K dataset')
    if len(manifest)!=3749 or len({r['image'] for r in manifest})!=3749:
        raise ValueError('Prepared test manifest is incomplete or duplicated')
    if len({r['video_id'] for r in manifest})!=8:
        raise ValueError('Expected exactly eight test video groups')
    if sum(int(r['num_yolo_labels']) for r in manifest)!=62015:
        raise ValueError('Test annotation total changed')
    for r in manifest:
        if step11f.seq_of(r['image']).removeprefix('video-')!=r['video_id']:
            raise ValueError(f'Image/video association mismatch: {r["image"]}')
    return True


def select_test_rows(manifest:list[dict],limit:int)->list[dict]:
    """For smoke, sample one image from EACH of eight videos; full=all 3749."""
    if not isinstance(limit,int) or limit<0 or (limit!=0 and limit<8):
        raise ValueError('Use --limit 0 for all images, or >=8 to span eight video groups')
    rows=sorted(manifest,key=lambda r:r['image'])
    if not limit or limit>=len(rows):
        return rows
    chosen=[];videos=set()
    for r in rows:
        if r['video_id'] not in videos:
            chosen.append(r);videos.add(r['video_id'])
    if len(videos)!=8:
        raise ValueError('Missing one or more video groups')
    selected={r['image'] for r in chosen}
    for r in rows:
        if len(chosen)>=limit: break
        if r['image'] not in selected:
            chosen.append(r);selected.add(r['image'])
    return sorted(chosen,key=lambda r:r['image'])


def validate_dev_references(c:dict,base_path:Path,base:dict,g_path:Path,gcfg:dict):
    """Validate Step11G cross-run reference and frozen checkpoint provenance."""
    from prepare_development_split import sha256
    a=Path(c['step11g_reference_a']);b=Path(c['step11g_verified_b'])
    aa,ad,af=step11g.load_comparison(a)
    bb,bd,bf=step11g.load_comparison(b)
    if not bb.get('cross_container_verified') or not bb.get('repeats_verified'):
        raise RuntimeError('Step11G B has not verified the full reference')
    if aa['images']!=200 or bb['images']!=200 or aa['sequences']!=17 or bb['sequences']!=17:
        raise RuntimeError('Step11G reference is not the full 200-image experiment')
    mismatch=step11g.compare_run(bb,bd,bf,aa,ad,af)
    if mismatch:
        raise RuntimeError(f'Step11G A/B reference disagrees: {mismatch[0]}')
    if sha256(base_path)!=bb['base_config_sha256'] or sha256(g_path)!=bb['config_sha256']:
        raise RuntimeError('Step11G reference configurations changed')
    sr_path=Path(c['large_sr_checkpoint']);det=Path(base['detector_checkpoint'])
    if sha256(sr_path)!=bb['large_sr_sha256'] or sha256(det)!=bb['detector_sha256']:
        raise RuntimeError('Frozen model checkpoint hash differs from Step11G')
    return bb, {'step11g_a_sha256':sha256(a),'step11g_b_sha256':sha256(b),
                'large_sr_sha256':sha256(sr_path),'detector_sha256':sha256(det),
                'base_config_sha256':sha256(base_path),'step11g_config_sha256':sha256(g_path)}


def write_csv(path:Path,rows:list[dict]):
    if not rows or path.exists():
        raise ValueError('Refusing empty/overwritten CSV output')
    import csv
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def validate_per_image(detections:list[dict],fingerprints:list[dict],trials:list[dict],nimages:int,nrepeats:int):
    identities={(r['image'],r['method'],float(r['threshold'])) for r in detections}
    fingerprints_id={(r['image'],r['method']) for r in fingerprints}
    timing_id={(r['image'],r['method'],int(r['repeat'])) for r in trials}
    if (len(identities)!=nimages*len(METHODS)*len(CONFIDENCES) or len(identities)!=len(detections)
        or len(fingerprints_id)!=nimages*len(METHODS) or len(fingerprints_id)!=len(fingerprints)
        or len(timing_id)!=nimages*len(METHODS)*nrepeats or len(timing_id)!=len(trials)):
        raise RuntimeError('Incomplete/duplicate test records')
    counts={}
    for r in detections:
        key=(r['image'],r['method'])
        if float(r['threshold'])==0.25:
            counts[key]=r
        if r['method'] not in METHODS or float(r['threshold']) not in CONFIDENCES:
            raise ValueError('Unexpected test detection method/threshold')
        if not (0<=int(r['small_tp'])<=int(r['small_gt']) and 0<=int(r['all_tp']) and 0<=int(r['fp'])):
            raise ValueError('Invalid test metrics')
    for r in detections:
        if float(r['threshold'])==0.40:
            ref=counts[r['image'],r['method']]
            if r['small_tp']>ref['small_tp'] or r['all_tp']>ref['all_tp'] or r['fp']>ref['fp']:
                raise ValueError('Raising the post-NMS threshold unexpectedly increased counts')
    return True


def comparison(dets:list[dict],a:str,ta:float,b:str,tb:float,penalty:float,seed:int,n_boot:int)->dict:
    by={(r['image'],r['method'],float(r['threshold'])):r for r in dets}
    sequences=defaultdict(list)
    for r in dets:
        if r['method']==a and float(r['threshold'])==ta:
            sequences[r['sequence']].append(r['image'])
    diffs={};dtp=dfp=0
    for seq,images in sequences.items():
        for im in images:
            ra=by[im,a,ta];rb=by[im,b,tb]
            if ra['small_gt']!=rb['small_gt']:
                raise ValueError('Different ground-truth counts across methods')
            t=int(ra['small_tp'])-int(rb['small_tp'])
            f=int(ra['fp'])-int(rb['fp'])
            diffs[im]=t-penalty*f
            dtp+=t;dfp+=f
    ci=step11f.cluster_ci(diffs,sequences,seed,n_boot)
    return {'a':{'method':a,'confidence':ta},'b':{'method':b,'confidence':tb},
            'delta_small_tp':dtp,'delta_fp':dfp,
            'delta_utility_total':round(dtp-penalty*dfp,6),
            'delta_utility_per_image':round(sum(diffs.values())/len(diffs),6),
            'sequence_bootstrap_95ci_per_image_exploratory':ci,
            'image_wins_ties_losses':{'wins':sum(v>0 for v in diffs.values()),
                                      'ties':sum(v==0 for v in diffs.values()),
                                      'losses':sum(v<0 for v in diffs.values())},
            'sequence_wins_ties_losses':{
                'wins':sum(sum(diffs[i] for i in ims)>0 for ims in sequences.values()),
                'ties':sum(sum(diffs[i] for i in ims)==0 for ims in sequences.values()),
                'losses':sum(sum(diffs[i] for i in ims)<0 for ims in sequences.values())},
            'num_sequences':len(sequences)}


def summarize(dets:list[dict],timings:list[dict],penalty:float,seed:int,n_boot:int)->dict:
    methods={}
    for t in CONFIDENCES:
        block={}
        for m in METHODS:
            subset=[r for r in dets if r['method']==m and float(r['threshold'])==t]
            if not subset:
                raise ValueError('Missing confidence/method results')
            totals={k:sum(int(r[k]) for r in subset) for k in KEYS}
            lat=defaultdict(list)
            for r in timings:
                if r['method']==m:
                    lat[r['image']].append(float(r['total_ms']))
            if set(lat)!={r['image'] for r in subset}:
                raise ValueError('Missing per-image timing')
            totals['small_recall']=round(totals['small_tp']/totals['small_gt'],6) if totals['small_gt'] else None
            totals['absolute_utility']=round(totals['small_tp']-penalty*totals['fp'],6)
            totals['mean_image_median_total_ms']=round(statistics.mean(statistics.median(x) for x in lat.values()),6)
            block[m]=totals
        methods[f'{t:.2f}']=block
    planned={'primary_contrast025_vs_bicubic025':PRIMARY,
             'secondary_contrast040_vs_bicubic040':SECONDARY[0],
             'secondary_contrast040_vs_bicubic025':SECONDARY[1],
             'secondary_random025_vs_bicubic025':('random4_bicubic_first_h10',0.25,'full_bicubic_640',0.25),
             'secondary_full_fsrcnn025_vs_bicubic025':('full_large_fsrcnn_640',0.25,'full_bicubic_640',0.25)}
    comparisons={k:comparison(dets,*pair,penalty,seed+i*29,n_boot) for i,(k,pair) in enumerate(planned.items())}
    return {'by_confidence':methods,'predeclared_comparisons':comparisons,
            'metric':'small-object TP minus 0.25*FP; class-aware matching IoU >=0.50; conf is post-NMS cutoff'}


def load_comparison(path:Path):
    import csv
    if path.name!='summary.json' or not path.is_file():
        raise FileNotFoundError('Cross-run reference must be an existing Step11L summary.json')
    summary=json.loads(path.read_text(encoding='utf-8'))
    if summary.get('protocol')!=PROTOCOL or not summary.get('repeats_verified'):
        raise ValueError('Reference has wrong protocol or repeat mismatches')
    def load(name):
        with (path.parent/name).open('r',newline='',encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))
    d=load('detection_per_image_thresholds.csv')
    f=load('fingerprints_per_image.csv')
    expect=summary['images']*len(METHODS)
    if len(d)!=expect*len(CONFIDENCES) or len(f)!=expect:
        raise ValueError('Reference test CSV is incomplete')
    return summary,d,f


def compare_run(current:dict,dets:list[dict],fps:list[dict],ref:dict,refd:list[dict],reff:list[dict]):
    if current.get('protocol')!=PROTOCOL or ref.get('protocol')!=PROTOCOL:
        raise ValueError('Cross-run protocol mismatch')
    provenance=('config_sha256','step11g_provenance','step11k_manifest_content_sha256',
                'step11k_summary_sha256','environment','deterministic_settings','images',
                'sequences','repeats_per_image','methods','thresholds','prepared_image_names_sha256')
    for k in provenance:
        if current.get(k)!=ref.get(k):
            raise ValueError(f'Cross-run test provenance mismatch: {k}')
    def convert(rows,fields):
        keys=set(); normalized=[]
        for r in rows:
            key=(str(r['image']),str(r['method']),str(r.get('threshold','')))
            if key in keys: raise ValueError(f'Duplicate comparison row {key}')
            keys.add(key)
            normalized.append({**r,'method':r['method']+('::'+str(r['threshold']) if 'threshold' in r else '')})
        return normalized
    changes=step11g.compare_rows(convert(dets,KEYS),convert(refd,KEYS),KEYS)
    changes+=step11g.compare_rows(fps,reff,FP_FIELDS)
    return changes


def run(args):
    if args.out.exists() or args.limit<0 or not 3<=args.repeats<=5:
        raise ValueError('Use a NEW report folder; --limit >=0 and repeats 3..5')
    if args.compare and args.compare.parent.resolve()==args.out.resolve():
        raise ValueError('Cannot compare a run to itself')
    flags=step11g.setup_determinism()
    from prepare_development_split import sha256
    from evaluate_selective_sr import choose_tiles
    from probe_roi_utility import roi_grid,prediction_records,detection_counts
    from evaluate_bicubic_baseline import ground_truth
    from benchmark_selective_fsrcnn import load_sr_checkpoint
    from utitherm.sr.bicubic_first_fsrcnn_step11a import render_bicubic_first_fsrcnn
    from utitherm.sr.roi_fsrcnn import render_fsrcnn

    c=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    base_path=Path(c['base_development_config']);base=yaml.safe_load(base_path.read_text(encoding='utf-8'))
    g_path=Path(c['step11g_config']);gcfg=yaml.safe_load(g_path.read_text(encoding='utf-8'))
    k_path=Path(c['step11k_config']);kcfg=yaml.safe_load(k_path.read_text(encoding='utf-8'))
    validate_frozen_config(c,base,gcfg,kcfg)
    gref,gsha=validate_dev_references(c,base_path,base,g_path,gcfg)
    k_report=Path(c['step11k_report']);k_data=Path(c['prepared_dataset'])
    source_summary,manifest=step11k.load_outputs_manifest(k_report)
    validate_sealed_test(c,source_summary,manifest,full=not bool(args.limit))
    step11k.verify_outputs(k_data,k_report,check_all=True)
    # The source header/provenance remains sealed; the prepared images are checked
    # individually using Step11K per-file SHA256 regardless of smoke/full mode.
    selected=select_test_rows(manifest,args.limit)
    if len({r['video_id'] for r in selected})!=8:
        raise ValueError('At least one sample from each of 8 videos required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for reproducible evaluation/timing')
    device=torch.device('cuda:0')
    sr=load_sr_checkpoint(Path(c['large_sr_checkpoint']),device)
    from ultralytics import YOLO
    detector=YOLO(str(base['detector_checkpoint']))
    env=step11g.env_snapshot()
    if env!=gref['environment']:
        raise RuntimeError('Environment not identical to verified Step11G reference')
    tiles=roi_grid(640,512,4,4)
    if len(tiles)!=16: raise RuntimeError('Grid is not 4x4')
    dummy=np.zeros(step11f.SHAPE,np.uint8)
    for _ in range(2):
        for method in METHODS:
            chosen=[0,5,10,15] if '_first_h10' in method else []
            rendered,_=step11f.render(method,dummy,tiles,chosen,sr,device,render_bicubic_first_fsrcnn,render_fsrcnn)
            prediction_records(detector,rendered,base)
    step11f.synchronize(device)

    trials=[];dets=[];fps=[];repeat_changes=[]
    for i,row in enumerate(selected,1):
        im=row['image'];vid=row['video_id']
        gray=cv2.imread(str(k_data/'images'/'test'/im),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=step11f.SHAPE or gray.dtype!=np.uint8:
            raise ValueError(f'Invalid prepared test LR PNG {im}')
        gts=ground_truth(k_data/'labels'/'test'/(Path(im).stem+'.txt'))
        if len(gts)!=int(row['num_yolo_labels']):
            raise RuntimeError(f'Unexpected test label count {im}')
        picks={m:tuple(step11f.selected_tiles(m,gray,im,tiles,int(base['seed']),choose_tiles)) for m in METHODS}
        first={};one_fps={}
        for rep in range(args.repeats):
            order=list(METHODS)
            random.Random(int(base['seed'])+i*1073+rep).shuffle(order)
            for m in order:
                timing,preds,finger=step11g.render_detect_trial(
                    m,gray,im,tiles,int(base['seed']),sr,detector,base,device,
                    choose_tiles,render_bicubic_first_fsrcnn,render_fsrcnn,prediction_records,picks)
                trials.append({'image':im,'sequence':'video-'+vid,'method':m,'repeat':rep,
                               **{k:round(float(v),6) for k,v in timing.items()}})
                if rep==0:
                    first[m]=preds;one_fps[m]=finger
                else:
                    for field in FP_FIELDS:
                        if str(finger[field])!=str(one_fps[m][field]):
                            repeat_changes.append({'image':im,'method':m,'repeat':rep,'field':field,
                                                   'reference':str(one_fps[m][field]),'current':str(finger[field])})
        for m in METHODS:
            fps.append({'image':im,'sequence':'video-'+vid,'method':m,**one_fps[m]})
            for t in CONFIDENCES:
                counts=detection_counts(gts,first[m],t)
                dets.append({'image':im,'sequence':'video-'+vid,'method':m,'threshold':f'{t:.2f}',**counts})
        if i==1 or i%100==0 or i==len(selected):
            print(f'Step11L deterministic disjoint TEST evaluated {i}/{len(selected)} frames',flush=True)
    validate_per_image(dets,fps,trials,len(selected),args.repeats)
    stats=summarize(dets,trials,0.25,int(c['bootstrap_seed']),int(c['bootstrap_iterations']))
    report={'task':'Step11L: first locked video-disjoint FLIR TEST evaluation; not cross-dataset',
            'created_utc':datetime.now(timezone.utc).isoformat(),'protocol':PROTOCOL,
            'config_sha256':sha256(args.config),
            'step11g_provenance':gsha,
            'step11k_manifest_content_sha256':source_summary['manifest_content_sha256'],
            'step11k_summary_sha256':sha256(k_report/'summary.json'),
            'prepared_image_names_sha256':hashlib.sha256(json.dumps([r['image'] for r in selected],separators=(',',':')).encode()).hexdigest(),
            'environment':env,'deterministic_settings':flags,
            'images':len(selected),'sequences':len({r['video_id'] for r in selected}),
            'repeats_per_image':args.repeats,'methods':list(METHODS),
            'thresholds':list(CONFIDENCES),
            'repeats_verified':not repeat_changes,
            'cross_container_reference':str(args.compare) if args.compare else None,
            'cross_container_verified':False,'cross_run_mismatch_count':None,
            'test_performance_was_not_used_for_selection':True,
            'predeclared_primary_comparison':'contrast4@0.25 versus bicubic@0.25',
            'predeclared_secondary_comparisons':['contrast4@0.40 versus bicubic@0.40','contrast4@0.40 versus bicubic@0.25'],
            'results':stats,
            'limitations':[
              'This is the first declared test split from FLIR ADAS v2, not a different dataset/camera.',
              'Eight independent VIDEO groups; individual frames within each group are correlated.',
              'Conf 0.40 was identified on development data, not independently selected from TEST.',
              'Confidence 0.40 is post-NMS filtering of YOLO outputs inferred with conf 0.25; not rerun at conf 0.40.',
              'IoU 0.50 class-aware small TP/FP utility is a project diagnostic, not standard mAP.',
              'Excluded COCO categories can appear in scenes; some detections may count as FPs.',
              'All four methods use identical trained models and 8-bit thermal-to-LR pipeline.',
              'Single random-4 seed; do not claim robustness over random selector seeds.',
              'Latency uses deterministic kernels and GPU sync, excludes disk IO/model loading.',
              'Confidence/selection must NOT be changed based on this test outcome.',
            ]}
    diffs=[]
    if args.compare:
        ref,rd,rf=load_comparison(args.compare)
        diffs=compare_run(report,dets,fps,ref,rd,rf)
        report['cross_run_mismatch_count']=len(diffs)
        report['cross_container_verified']=not diffs and not repeat_changes
    args.out.mkdir(parents=True)
    write_csv(args.out/'detection_per_image_thresholds.csv',dets)
    write_csv(args.out/'fingerprints_per_image.csv',fps)
    write_csv(args.out/'timing_per_trial.csv',trials)
    if repeat_changes:write_csv(args.out/'repeat_mismatches.csv',repeat_changes)
    if diffs:write_csv(args.out/'cross_run_mismatches.csv',diffs)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('\nSTEP11L — frozen VIDEO-DISJOINT test results (DO NOT tune from these):')
    for t in CONFIDENCES:
        print(f'  Post-NMS conf={t:.2f}')
        for m in METHODS:
            r=stats['by_confidence'][f'{t:.2f}'][m]
            print(f'    {m:32} smallTP={r["small_tp"]:5} FP={r["fp"]:5} U={r["absolute_utility"]:+.2f} latency={r["mean_image_median_total_ms"]:.3f}ms')
    for name,result in stats['predeclared_comparisons'].items():
        print(f'  {name}: ΔTP={result["delta_small_tp"]:+d} ΔFP={result["delta_fp"]:+d} '
              f'ΔU={result["delta_utility_total"]:+.2f} CI/image={result["sequence_bootstrap_95ci_per_image_exploratory"]}')
    print(f'SAVED: {args.out/"summary.json"}')
    if repeat_changes or diffs:
        raise RuntimeError(f'Step11L deterministic reproducibility FAILED: repeat={len(repeat_changes)}, cross_run={len(diffs)}; outputs retained')
    print('VERIFIED: all within-run repeat fingerprints match; '+
          ('cross-container test fingerprints/counts agree.' if args.compare else 'A run sealed; run fresh B with --compare.'))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/flir_disjoint_test_evaluation_step11l.yaml'))
    parser.add_argument('--limit',type=int,default=8,help='0=all 3749 test images; 8=one per test video')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--compare',type=Path,default=None,help='Step11L reference summary.json from fresh other container')
    run(parser.parse_args())

if __name__=='__main__':
    main()
