"""Step 11G: deterministic, ACTUAL combined-four-region detector evaluation.

Retrospective FLIR development ONLY (not independent test).
Requires CUBLAS_WORKSPACE_CONFIG=:4096:8 in the process environment.
Examples:
  python scripts/evaluate_bicubic_first_deterministic_step11g.py --limit 8 --out reports/development/step11g_smoke_a
  python scripts/evaluate_bicubic_first_deterministic_step11g.py --limit 8 --out reports/development/step11g_smoke_b --compare reports/development/step11g_smoke_a/summary.json

Step10D historical detection counts CANNOT be exact cross-protocol control here;
Step11G verifies frozen provenance and compares same-protocol A/B image and
prediction fingerprints, with unchanged strict Step11F checks untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_bicubic_first_combined_step11f as prev

PROTOCOL='STEP11G_DETERMINISTIC_BICUBIC_FIRST_4ROI_FLIR_DEV_V1'
REQUIRED_CUBLAS=':4096:8'
METRICS=('small_gt','small_tp','all_tp','fp')


def setup_determinism(env=None):
    """Fail closed if cuBLAS environment is missing, before CUDA initialization."""
    env=os.environ if env is None else env
    if env.get('CUBLAS_WORKSPACE_CONFIG')!=REQUIRED_CUBLAS:
        raise RuntimeError('Step11G requires CUBLAS_WORKSPACE_CONFIG=:4096:8 '
                           '(pass -e CUBLAS_WORKSPACE_CONFIG=:4096:8 to docker compose run)')
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was initialized before Step11G deterministic setup')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    if (not torch.are_deterministic_algorithms_enabled() or
        torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic or
        torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32):
        raise RuntimeError('Deterministic GPU settings did not take effect')
    return {'CUBLAS_WORKSPACE_CONFIG':REQUIRED_CUBLAS,
            'torch_deterministic_algorithms':True, 'cudnn_benchmark':False,
            'cudnn_deterministic':True,'cudnn_allow_tf32':False,
            'matmul_allow_tf32':False,'opencv_threads':1}


def digest_bytes(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def image_fingerprint(image:np.ndarray) -> str:
    if image.shape!=(512,640) or image.dtype!=np.uint8 or image.ndim!=2:
        raise ValueError('Cannot fingerprint non-uint8 512x640 grayscale enhanced image')
    return digest_bytes(image.tobytes(order='C'))


def prediction_fingerprint(preds:list[dict])->str:
    """Exact normalized JSON float serialization; equality is deliberately strict."""
    if not isinstance(preds,list):
        raise ValueError('Predictions must be a list')
    records=[]
    for p in preds:
        if not isinstance(p,dict) or set(p)!={'box','conf','cls'} or len(p['box'])!=4:
            raise ValueError('Unexpected detector prediction schema')
        box=list(map(float,p['box']))
        c=float(p['conf'])
        if not all(np.isfinite(v) for v in (*box,c)) or not 0<=c<=1:
            raise ValueError('Nonfinite or invalid prediction')
        records.append({'box':box,'conf':c,'cls':int(p['cls'])})
    return digest_bytes(json.dumps(records,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8'))


def identity(row:dict)->tuple[str,str]:
    return str(row['image']),str(row['method'])


def compare_rows(current:list[dict],reference:list[dict],fields:tuple[str,...]) -> list[dict]:
    """Explicit row identity+unique checks; return every mismatching image/method."""
    def as_map(rows,label):
        result={}
        for r in rows:
            k=identity(r)
            if k in result:
                raise ValueError(f'Duplicate {label} row {k}')
            result[k]=r
        return result
    a=as_map(current,'current');b=as_map(reference,'reference')
    if not a or set(a)!=set(b):
        raise ValueError(f'Image/method records disagree: missing={sorted(set(b)-set(a))[:3]}, '
                         f'extra={sorted(set(a)-set(b))[:3]}')
    mismatches=[]
    for k in sorted(a):
        for col in fields:
            if str(a[k][col])!=str(b[k][col]):
                mismatches.append({'image':k[0],'method':k[1],'field':col,
                                   'reference':str(b[k][col]),'current':str(a[k][col])})
    return mismatches


def compare_run(current_summary:dict, current_dets:list[dict], current_fps:list[dict],
                reference_summary:dict, reference_dets:list[dict], reference_fps:list[dict]):
    """Require exact per-image/render/pred/selection and matching checkpoint protocol."""
    if reference_summary.get('protocol')!=PROTOCOL or current_summary.get('protocol')!=PROTOCOL:
        raise ValueError('Cross-run protocol mismatch')
    for key in ('config_sha256','base_config_sha256','manifest_sha256',
                'large_sr_sha256','detector_sha256','frozen_checkpoint_sha256',
                'environment','images','sequences','repeats_per_image','methods',
                'deterministic_settings'):
        if current_summary.get(key)!=reference_summary.get(key):
            raise ValueError(f'Cross-run provenance/config mismatch: {key}')
    if not reference_summary.get('repeats_verified'):
        raise ValueError('Reference is not an internally verified Step11G run')
    det_diffs=compare_rows(current_dets,reference_dets,METRICS)
    fps_diffs=compare_rows(current_fps,reference_fps,
                           ('selected_tiles','rendered_sha256','predictions_sha256','num_predictions'))
    return det_diffs+fps_diffs


def csv_rows(path:Path)->list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('r',encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def load_comparison(path:Path):
    if path.name!='summary.json' or not path.is_file():
        raise FileNotFoundError('Comparison must name existing Step11G summary.json')
    reference=json.loads(path.read_text(encoding='utf-8'))
    dets=csv_rows(path.with_name('detection_per_image.csv'))
    fps=csv_rows(path.with_name('fingerprints_per_image.csv'))
    if len(dets)!=int(reference.get('images',-1))*len(prev.MODES) or len(fps)!=len(dets):
        raise ValueError('Reference CSV completeness check failed')
    return reference,dets,fps


def env_snapshot():
    from ultralytics import __version__ as ultra_version
    return {'python':platform.python_version(),'pytorch':torch.__version__,
            'torch_cuda':torch.version.cuda,'cudnn':str(torch.backends.cudnn.version()),
            'gpu':torch.cuda.get_device_name(0),'opencv':cv2.__version__,
            'numpy':np.__version__,'ultralytics':ultra_version}


def render_detect_trial(method, gray, image, tiles, seed, sr, detector, base, device,
                        choose_tiles, combined_renderer, full_renderer, predict_fn, frozen_selection):
    prev.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0=time.perf_counter_ns()
    chosen=prev.selected_tiles(method,gray,image,tiles,seed,choose_tiles)
    if tuple(chosen)!=tuple(frozen_selection[method]):
        raise ValueError('Nondeterministic region selection')
    prev.synchronize(device)
    t1=time.perf_counter_ns()
    output,pixels=prev.render(method,gray,tiles,chosen,sr,device,combined_renderer,full_renderer)
    prev.synchronize(device)
    t2=time.perf_counter_ns()
    preds=predict_fn(detector,output,base)
    prev.synchronize(device)
    t3=time.perf_counter_ns()
    # Hashes calculated AFTER the synchronized timing window, to keep runtime
    # comparable to Step11F instead of charging SHA256 cost to the detector.
    pixhash=image_fingerprint(output)
    predhash=prediction_fingerprint(preds)
    timing={'selector_ms':(t1-t0)/1e6,'sr_ms':(t2-t1)/1e6,
            'detector_ms':(t3-t2)/1e6,'total_ms':(t3-t0)/1e6,
            'computed_pixels':pixels,
            'peak_torch_mib':torch.cuda.max_memory_allocated(device)/1024**2}
    return timing,preds,{'rendered_sha256':pixhash,'predictions_sha256':predhash,
                         'num_predictions':len(preds),'selected_tiles':json.dumps(chosen,separators=(',',':'))}


def run(args):
    if args.limit<0 or not 3<=args.repeats<=10:
        raise ValueError('limit must be nonnegative and repeats 3..10')
    if args.out.exists():
        raise FileExistsError(f'Output folder must be new: {args.out}')
    if args.compare is not None and args.out.resolve()==args.compare.parent.resolve():
        raise ValueError('Cannot compare a report against itself')
    flags=setup_determinism()
    from prepare_development_split import sequence_id,training_sequences,check_manifest,sha256
    from evaluate_selective_sr import choose_tiles
    from probe_roi_utility import roi_grid,prediction_records,detection_counts
    from evaluate_bicubic_baseline import ground_truth
    from benchmark_selective_fsrcnn import load_sr_checkpoint
    from utitherm.sr.bicubic_first_fsrcnn_step11a import render_bicubic_first_fsrcnn
    from utitherm.sr.roi_fsrcnn import render_fsrcnn

    experiment=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    base_path=Path(experiment['base_development_config'])
    base=yaml.safe_load(base_path.read_text(encoding='utf-8'))
    manifest_path=Path(experiment['frozen_manifest'])
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    prev.validate_protocol(experiment,base,manifest)
    hist_path=Path(experiment['reference_step10d_summary'])
    hist=json.loads(hist_path.read_text(encoding='utf-8'))
    sr_path=Path(experiment['large_sr_checkpoint'])
    frozen_shas={k:sha256(Path(base[k])) for k in ('predictor_checkpoint','detector_checkpoint','sr_checkpoint')}
    for k in frozen_shas:
        if frozen_shas[k]!=manifest['checkpoint_sha256'][k]:
            raise ValueError(f'Frozen checkpoint mismatch: {k}')
    prev.verify_reference_provenance(hist,sha256(manifest_path),sha256(sr_path),frozen_shas)
    val=Path(base['eval_lr']); train=Path(base['train_lr']);labels=val.parent.parent/'labels'/'val'
    trseq=training_sequences(train,Path(base['utility_csv']),Path(base['split_manifest']))
    names=check_manifest(manifest,val,labels,trseq,Path(base['utility_csv']),Path(base['split_manifest']))
    if len(names)!=200 or len({sequence_id(i) for i in names})!=17:
        raise ValueError('Frozen development manifest mismatch')
    if args.limit:
        names=names[:args.limit]
    if len(names)<2 or len({sequence_id(i) for i in names})<2:
        raise ValueError('Need at least 2 images and 2 sequences')
    if not torch.cuda.is_available():
        raise RuntimeError('Step11G needs CUDA for same-hardware A/B comparison')
    device=torch.device('cuda:0')
    sr=load_sr_checkpoint(sr_path,device)
    from ultralytics import YOLO
    detector=YOLO(str(base['detector_checkpoint']))
    env=env_snapshot()
    tiles=roi_grid(640,512,4,4)
    if len(tiles)!=16:
        raise ValueError('Unexpected ROI grid')
    dummy=np.zeros(prev.SHAPE,np.uint8)
    for _ in range(2):
        for mode in prev.MODES:
            chosen=[0,5,10,15] if '_first_h10' in mode else []
            frame,_=prev.render(mode,dummy,tiles,chosen,sr,device,
                                render_bicubic_first_fsrcnn,render_fsrcnn)
            prediction_records(detector,frame,base)
    prev.synchronize(device)

    trials=[];dets=[];fingerprints=[];repeat_mismatches=[]
    for index,image in enumerate(names,1):
        gray=cv2.imread(str(val/image),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=prev.SHAPE or gray.dtype!=np.uint8:
            raise ValueError(f'Unexpected input frame: {image}')
        picks={m:tuple(prev.selected_tiles(m,gray,image,tiles,int(base['seed']),choose_tiles)) for m in prev.MODES}
        first={}; per_mode_first={}
        for repeat in range(args.repeats):
            order=list(prev.MODES)
            random.Random(int(base['seed'])+index*1073+repeat).shuffle(order)
            for mode in order:
                timing,preds,fp=render_detect_trial(mode,gray,image,tiles,int(base['seed']),sr,detector,base,
                                device,choose_tiles,render_bicubic_first_fsrcnn,render_fsrcnn,
                                prediction_records,picks)
                trials.append({'image':image,'sequence':sequence_id(image),'repeat':repeat,'method':mode,
                               **{k:round(float(v),6) for k,v in timing.items()}})
                if repeat==0:
                    first[mode]=preds
                    per_mode_first[mode]=fp
                else:
                    for field in ('rendered_sha256','predictions_sha256','selected_tiles','num_predictions'):
                        if fp[field]!=per_mode_first[mode][field]:
                            repeat_mismatches.append({'image':image,'method':mode,'repeat':repeat,
                                'field':field,'reference':str(per_mode_first[mode][field]),
                                'current':str(fp[field])})
        gts=ground_truth(labels/(Path(image).stem+'.txt'))
        for mode in prev.MODES:
            dets.append({'image':image,'sequence':sequence_id(image),'method':mode,
                         **detection_counts(gts,first[mode],float(base['confidence']))})
            fingerprints.append({'image':image,'sequence':sequence_id(image),'method':mode,
                                 **per_mode_first[mode]})
        if index==1 or index%10==0 or index==len(names):
            print(f'Deterministic Step11G evaluated {index}/{len(names)} frames',flush=True)

    stats=prev.summarize(dets,trials,args.repeats,float(experiment['false_positive_penalty']),
                         int(experiment['bootstrap_seed']),int(experiment['bootstrap_iterations']))
    report={'task':'Step11G: actual four-region Bicubic-first in deterministic GPU mode',
            'created_utc':datetime.now(timezone.utc).isoformat(),'protocol':PROTOCOL,
            'config_sha256':sha256(args.config),'base_config_sha256':sha256(base_path),
            'manifest_sha256':sha256(manifest_path),'large_sr_sha256':sha256(sr_path),
            'detector_sha256':frozen_shas['detector_checkpoint'],
            'frozen_checkpoint_sha256':frozen_shas,
            'history_step10d_sha256':sha256(hist_path),
            'historical_step10d_counts_are_not_exact_reference':True,
            'deterministic_settings':flags,'environment':env,
            'images':len(names),'sequences':len({sequence_id(i) for i in names}),
            'repeats_per_image':args.repeats,'methods':list(prev.MODES),
            'repeats_verified':not repeat_mismatches,
            'cross_container_reference':str(args.compare) if args.compare else None,
            'cross_container_verified':False,'cross_run_mismatch_count':None,
            'results':stats,
            'limitations':[
                 'Frozen FLIR validation development sample is not independent/untouched.',
                 'Detector diagnostics conf .25 IoU .50 are not standard mAP.',
                 'Reference Step10D ran nondeterministically; do not demand numeric identity across protocols.',
                 'Repeat fingerprint checks include SR and exact YOLO floating-point predictions.',
                 'SHA fingerprints are computed after timed inference; deterministic kernels may affect latency.',
                 'Random-four is a single frozen seed; cluster bootstrap with only 17 sequences is exploratory.',
             ]}
    cross_diffs=[]
    if args.compare is not None:
        ref,rd,rf=load_comparison(args.compare)
        cross_diffs=compare_run(report,dets,fingerprints,ref,rd,rf)
        report['cross_run_mismatch_count']=len(cross_diffs)
        report['cross_container_verified']=not cross_diffs and not repeat_mismatches
    args.out.mkdir(parents=True)
    prev.write_csv(args.out/'detection_per_image.csv',dets)
    prev.write_csv(args.out/'timing_per_trial.csv',trials)
    prev.write_csv(args.out/'fingerprints_per_image.csv',fingerprints)
    if repeat_mismatches:
        prev.write_csv(args.out/'repeat_mismatches.csv',repeat_mismatches)
    if cross_diffs:
        prev.write_csv(args.out/'cross_run_mismatches.csv',cross_diffs)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('\nSTEP11G — deterministic four-method ACTUAL detection; previously reused FLIR DEV:')
    for mode in prev.MODES:
        m=stats['methods'][mode]
        print(f'  {mode:32} smallTP={m["small_tp"]:4} FP={m["fp"]:4} '
              f'recall={m["small_recall_conf_025"]:.3f} time={m["mean_image_median_total_ms"]:.3f} ms')
    for k,m in stats['comparisons'].items():
        print(f'  {k}: ΔTP={m["delta_small_tp"]:+d} ΔFP={m["delta_fp"]:+d} '
              f'ΔU={m["delta_utility_total"]:+.2f} '
              f'CI/image={m["sequence_bootstrap_utility_per_image_95ci"]}')
    print(f'SAVED — {args.out/"summary.json"}',flush=True)
    if repeat_mismatches or cross_diffs:
        raise RuntimeError(f'Deterministic reproducibility FAILED: repeat differences={len(repeat_mismatches)}, '
                           f'cross-run differences={len(cross_diffs)}. Saved diagnostics, do not use as verified.')
    if args.compare is None:
        print('REFERENCE CAPTURED; complete A/B verification by a fresh --compare run.')
    else:
        print('VERIFIED: repeat fingerprints and cross-run counts/render/prediction hashes agree.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/bicubic_first_deterministic_step11g.yaml'))
    p.add_argument('--limit',type=int,default=8,help='0=all 200 development images')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--compare',type=Path,default=None,help='Prior Step11G summary.json, created in another container')
    run(p.parse_args())

if __name__=='__main__':
    main()
