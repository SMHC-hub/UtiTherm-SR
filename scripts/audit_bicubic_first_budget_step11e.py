"""Step 11E: sequence-safe retrospective tile-budget/abstention audit.

Reuses EXACT Step 11D feature extraction and its previously fixed split.
No YOLO or FSRCNN inference. All results are SUMMED SINGLE-TILE utility proxies,
NOT the observed detection result of applying multiple tiles together.

Run: python scripts/audit_bicubic_first_budget_step11e.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

# Loading local sibling ensures the exact Step11D feature/scoring implementation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bicubic_first_predictability_step11d import (  # noqa: E402
    FEATURES, choose_indices, cluster_bootstrap_difference, extract_frame_features,
    sha256, validated_rows,
)

PROTOCOL = 'STEP_11E_BICUBIC_FIRST_BUDGET_RETROSPECTIVE_PROXY_V1'


def parse_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict) or cfg.get('scoring') != 'one_roi_marginal_utility_sum_proxy_ONLY':
        raise ValueError('Step11E configuration/protocol mismatch')
    budgets = cfg['budgets']
    if budgets != [0, 1, 2, 4] or cfg['gate_budgets'] != [0, 1, 2, 4]:
        raise ValueError('Budgets must remain the prespecified [0,1,2,4]')
    if cfg['policies'] != ['contrast', 'edge']:
        raise ValueError('Policies must remain [contrast,edge]')
    thresholds = cfg['contrast_gate_z_thresholds']
    if thresholds != [0.0,0.5,1.0,1.5,2.0]:
        raise ValueError('Gate thresholds changed; create a new protocol if changing them')
    if cfg['expected_tiles_per_image'] != 16 or cfg['fp_penalty'] != 0.25:
        raise ValueError('Original intervention protocol must be preserved')
    if int(cfg['bootstrap_iterations']) < 100:
        raise ValueError('Need >=100 bootstrap replicates')
    return cfg


def verify_sources(cfg: dict):
    csvfile = Path(cfg['source_csv'])
    srfile = Path(cfg['source_summary'])
    splitfile = Path(cfg['step11d_split'])
    step11d = Path(cfg['step11d_summary'])
    perimage = Path(cfg['step11d_test_per_image'])
    summary = json.loads(srfile.read_text(encoding='utf-8'))
    prior = json.loads(step11d.read_text(encoding='utf-8'))
    split = json.loads(splitfile.read_text(encoding='utf-8'))
    h = sha256(csvfile)
    if (summary.get('protocol') != 'STEP_11A_INCREMENTAL_LARGE_FSRCNN_OVER_BICUBIC_SINGLE_ROI_V1'
        or summary.get('baseline') != 'full_bicubic_640'
        or summary.get('roi_csv_sha256') != h
        or prior.get('source_csv_sha256') != h
        or split.get('csv_sha256') != h
        or prior.get('protocol') != 'STEP_11D_BICUBIC_FIRST_CHEAP_FEATURE_PREDICTABILITY_V1'):
        raise ValueError('Step11A/11D source hash or protocol mismatch')
    rows, grouped = validated_rows(csvfile,16,0.25)
    images = sorted(grouped)
    if (len(rows) != int(cfg['expected_interventions']) or
        len(images) != int(cfg['expected_images']) or
        len({r['sequence'] for r in rows}) != int(cfg['expected_sequences']) or
        int(summary.get('positive_utility',-1)) != sum(r['positive_int'] for r in rows) or
        set(images) != set(summary['source_provenance']['selected_images'])):
        raise ValueError('Source labels count/coverage inconsistent')
    groups = split['sequence_ids']
    by_split = split['images_by_split']
    if set(groups) != {'train','val','test'} or set(by_split) != set(groups):
        raise ValueError('Invalid Step11D split roles')
    allseqs = {r['sequence'] for r in rows}
    ids = [s for role in ('train','val','test') for s in groups[role]]
    if set(ids) != allseqs or len(ids) != len(allseqs):
        raise ValueError('Split sequences overlap or are incomplete')
    allimgs = [im for role in ('train','val','test') for im in by_split[role]]
    if set(allimgs) != set(images) or len(allimgs) != len(images):
        raise ValueError('Split images overlap or are incomplete')
    seq_by_im = {im:grouped[im][0]['sequence'] for im in images}
    for role in ('train','val','test'):
        if not by_split[role] or not groups[role]:
            raise ValueError(f'Empty role {role}')
        if any(seq_by_im[im] not in groups[role] for im in by_split[role]):
            raise ValueError('Image/sequence split mismatch')
        if (prior['train_val_test_image_counts'][role] != len(by_split[role]) or
            prior['train_val_test_sequence_counts'][role] != len(groups[role])):
            raise ValueError('Step11D summary split mismatch')
    return rows, grouped, by_split, seq_by_im, prior, perimage, {
        'source_csv_sha256': h,
        'source_summary_sha256': sha256(srfile),
        'step11d_split_sha256': sha256(splitfile),
        'step11d_summary_sha256': sha256(step11d),
        'step11d_test_per_image_sha256': sha256(perimage),
    }


def normalized_contrast(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores,dtype=float)
    if x.shape != (16,) or not np.isfinite(x).all():
        raise ValueError('Expected 16 finite tile contrast scores')
    std = float(np.std(x))
    return (x-float(np.mean(x))) / max(std,1e-8)


def picks(ranks: np.ndarray, budget: int, z_threshold: float | None = None) -> list[int]:
    if z_threshold is None:
        return choose_indices(ranks,budget)
    if not np.isfinite(z_threshold):
        raise ValueError('Nonfinite threshold')
    # Only regions with relatively above-average within-image contrast qualify.
    z = normalized_contrast(ranks)
    return choose_indices(z,budget,threshold=z_threshold)


def measure(images: list[str], utility: dict[str,np.ndarray],
            scores: dict[str,np.ndarray], budget: int,
            gate: float | None = None) -> dict:
    per_image = {}
    selected=0; pos=0; neg=0; abstain=0
    for im in images:
        idx = picks(scores[im],budget,gate)
        u = utility[im]
        per_image[im] = float(np.sum(u[idx]))
        selected += len(idx)
        pos += sum(float(u[i]) > 0 for i in idx)
        neg += sum(float(u[i]) < 0 for i in idx)
        abstain += int(len(idx)==0)
    return dict(mean_proxy_per_image=float(np.mean(list(per_image.values()))),
                per_image=per_image, selected_regions=selected,
                positive_selected=pos, negative_selected=neg, images_abstaining=abstain,
                mean_nominal_processed_lr_pixels_per_image=8400.0*selected/len(images))


def random_expected(images: list[str], utility: dict[str,np.ndarray], budget: int) -> dict:
    return dict(mean_proxy_per_image=float(np.mean([budget*np.mean(utility[im]) for im in images])),
                per_image={im:float(budget*np.mean(utility[im])) for im in images})


def select_gate(validation_images: list[str], utility: dict[str,np.ndarray],
                contrast: dict[str,np.ndarray], budgets: list[int], thresholds: list[float]) -> dict:
    """Predefined 21 policies: baseline 0 + 4*5? Actually 3*5 + baseline.

    Ties choose abstention (budget=0), then smaller budget, then higher threshold.
    No test label access allowed here by construction.
    """
    choices = []
    for k in budgets:
        for t in (thresholds if k > 0 else [None]):
            info = measure(validation_images,utility,contrast,k,t)
            mean=info['mean_proxy_per_image']
            # choose <= fewer enh when same utility; exact 0 favors k=0.
            choices.append((mean,-info['selected_regions'],-k,float(t) if t is not None else 1e9,k,t,info))
    best = max(choices,key=lambda x:x[:4])
    return dict(budget=best[4],z_threshold=best[5],
                validation_mean_proxy_per_image=best[0],
                validation_selected_regions=best[6]['selected_regions'],
                num_candidates=len(choices))


def bootstrap_comparison(a: dict,b: dict,seq: dict,seed: int,reps: int) -> dict:
    aa,bb = a['per_image'],b['per_image']
    delta={im:aa[im]-bb[im] for im in aa}
    ci=cluster_bootstrap_difference(aa,bb,seq,seed,reps)
    return dict(delta_proxy_per_image=float(np.mean(list(delta.values()))),
                sequence_bootstrap_exploratory_95ci=list(ci),
                image_wins_ties_losses=[sum(x>1e-12 for x in delta.values()),
                                        sum(abs(x)<=1e-12 for x in delta.values()),
                                        sum(x < -1e-12 for x in delta.values())],
                num_sequences=len(set(seq[im] for im in aa)))


def prior_step11d_test_alignment(images,utility,contrast,edge,prior_summary,prior_csv):
    """Fail closed if reconstructed Step11D 4-region decisions differ."""
    with prior_csv.open(newline='',encoding='utf-8-sig') as f:
        obs = list(csv.DictReader(f))
    if set(r['image'] for r in obs) != set(images) or len(obs) != len(images):
        raise ValueError('Prior Step11D per-image test rows do not align')
    for r in obs:
        im = r['image']
        expected = {
            'contrast_top4':measure([im],utility,contrast,4)['mean_proxy_per_image'],
            'edge_top4':measure([im],utility,edge,4)['mean_proxy_per_image'],
            'random4_expectation':float(4*np.mean(utility[im])),
            'first4':float(np.sum(utility[im][:4])),
            'bicubic_no_neural_sr':0.,
        }
        for key,v in expected.items():
            if key not in r or abs(float(r[key])-v)>1.1e-6:
                raise ValueError(f'Step11D decision replay differs for {im}/{key}: {r.get(key)} vs {v}')
    for kind in ('contrast','edge'):
        measured=measure(images,utility,contrast if kind=='contrast' else edge,4)['mean_proxy_per_image']
        prev=float(prior_summary['results']['test']['methods'][kind+'_top4']['mean_marginal_utility_per_image'])
        if abs(measured-prev)>1e-8:
            raise ValueError(f'Step11D {kind} aggregate replay differs')


def write_json(path:Path, data:dict):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def run(args: argparse.Namespace) -> dict:
    cfg=parse_config(args.config)
    rows,grouped,sets,seq_for_im,prior,prior_csv,hashes=verify_sources(cfg)
    # Read only VAL+TEST raw images. The TRAIN split is not used for model fitting.
    chosen=sorted(set(sets['val'])|set(sets['test']))
    util={}; contrast={}; edge={}
    lr_root=Path(cfg['train_lr'])
    cv2.setNumThreads(1)
    for n,im in enumerate(chosen,1):
        gray=cv2.imread(str(lr_root/im),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (256,320) or gray.dtype != np.uint8:
            raise ValueError(f'Missing or invalid LR thermal image: {lr_root/im}')
        rr=sorted(grouped[im],key=lambda r:r['tile_int'])
        rects=[tuple(int(r[k]) for k in ('tile_x0_hr','tile_y0_hr','tile_x1_hr','tile_y1_hr')) for r in rr]
        feat=extract_frame_features(gray,rects)
        util[im]=np.array([float(r['utility_float']) for r in rr])
        contrast[im]=feat[:,FEATURES.index('tile_std')]
        edge[im]=feat[:,FEATURES.index('tile_grad_mean')]
        if n==1 or n%25==0 or n==len(chosen):
            print(f'Reproduced Step11D features {n}/{len(chosen)} VAL+TEST frames',flush=True)
    prior_step11d_test_alignment(sets['test'],util,contrast,edge,prior,prior_csv)
    choice=select_gate(sets['val'],util,contrast,cfg['gate_budgets'],cfg['contrast_gate_z_thresholds'])
    report={}
    per_image=[]
    for role in ('val','test'):
        ims=sets[role]
        methods={}
        detailed={}
        for k in cfg['budgets']:
            methods[f'bicubic_budget0'] = measure(ims,util,contrast,0)
            if k>0:
                for policy in cfg['policies']:
                    ranks=contrast if policy=='contrast' else edge
                    methods[f'{policy}_top{k}']=measure(ims,util,ranks,k)
                methods[f'random{k}_expectation']=random_expected(ims,util,k)
                # Oracle is a retrospective diagnostic, not an achievable policy.
                methods[f'oracle_top{k}_hindsight']=measure(ims,util,util,k)
                # Optional-abstention oracle using actual single-intervention utilities.
                vals={};npos=0;sel=0; abstain=0
                for im in ims:
                    ids=choose_indices(util[im],k,threshold=0.)
                    vals[im]=float(np.sum(util[im][ids])); sel+=len(ids)
                    npos+=len(ids); abstain+=int(not ids)
                methods[f'oracle_gated{k}_hindsight']={'mean_proxy_per_image':float(np.mean(list(vals.values()))),
                    'per_image':vals,'selected_regions':sel,'positive_selected':npos,
                    'negative_selected':0,'images_abstaining':abstain,
                    'mean_nominal_processed_lr_pixels_per_image':8400.0*sel/len(ims)}
        ck,ct=choice['budget'],choice['z_threshold']
        methods['contrast_val_selected_gated']=measure(ims,util,contrast,ck,ct)
        comparisons={}
        if role=='test':
            base=methods['bicubic_budget0']
            for name,m in methods.items():
                if name=='bicubic_budget0' or name.startswith('oracle_'):
                    continue
                comparisons[name+'__vs__bicubic_budget0']=bootstrap_comparison(m,base,seq_for_im,
                    int(cfg['bootstrap_seed'])+31,int(cfg['bootstrap_iterations']))
            for k in (1,2,4):
                comparisons[f'contrast_top{k}__vs__random{k}_expectation']=bootstrap_comparison(
                    methods[f'contrast_top{k}'],methods[f'random{k}_expectation'],seq_for_im,
                    int(cfg['bootstrap_seed'])+31,int(cfg['bootstrap_iterations']))
        for im in ims:
            detail={'image':im,'sequence':seq_for_im[im],'split':role}
            for name,m in methods.items():detail[name]=float(m['per_image'][im])
            per_image.append(detail)
        report[role]={'num_images':len(ims),'num_sequences':len({seq_for_im[im] for im in ims}),
                      'methods':{name:{key:v for key,v in m.items() if key!='per_image'}
                                 for name,m in methods.items()},'comparisons':comparisons}
    out=Path(args.out) if args.out else Path(cfg['out'])
    if out.exists() and not args.overwrite:
        raise FileExistsError(f'Output exists: {out}; choose a fresh folder or use --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    names=sorted({k for r in per_image for k in r if k not in ('image','sequence','split')})
    with (out/'per_image_budget_proxy.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['image','sequence','split',*names],extrasaction='ignore')
        writer.writeheader()
        for row in per_image:writer.writerow(row)
    result={'task':'Step11E: Bicubic-first ROI budget/abstention; retrospective proxy ONLY',
            'protocol':PROTOCOL,'created_utc':datetime.now(timezone.utc).isoformat(),
            'sources_sha256':hashes,'config_sha256':sha256(args.config),
            'source_interventions':len(rows),'source_sequences':len({r['sequence'] for r in rows}),
            'source_images':len(grouped),'split_counts':{k:len(sets[k]) for k in sets},
            'validation_selected_contrast_gate':choice,
            'results':report,
            'limitations':[
                'Test video groups have been analyzed in Step11D; this is NOT a fresh final test.',
                'Detector and SR model both trained on this FLIR TRAIN source domain.',
                'Per-image scores sum separate single-tile outcomes; no multi-tile detection measured.',
                'Oracle uses hindsight labels; not achievable at inference or a combined-ROI bound.',
                'Test 5 sequence groups only; bootstrap CIs have high uncertainty.',
                'Full Bicubic is zero incremental neural SR by design, not zero latency.',
                'Computed LR pixels are NOMINAL (8400 per tile); no actual latency or memory measured.',
                'Validation-selected gate uses a small validation set; risk of overfitting selection.',
                'Because contrast-top4 was observed previously, follow-up budgets are exploratory.',
            ]}
    write_json(out/'summary.json',result)
    print('STEP 11E — fixed split, marginal-proxy results only (NOT observed combined ROI detection):')
    for name,m in report['test']['methods'].items():
        if not name.startswith('oracle_'):
            print(f'  {name:34s} {m["mean_proxy_per_image"]:+.4f}/image; selected={m.get("selected_regions","expected")}',flush=True)
    print(f'SAVED — {out/"summary.json"}; gate chosen using VAL only; TEST sequences={report["test"]["num_sequences"]}',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/bicubic_first_budget_step11e.yaml'))
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--overwrite',action='store_true')
    run(p.parse_args())

if __name__=='__main__':
    main()
