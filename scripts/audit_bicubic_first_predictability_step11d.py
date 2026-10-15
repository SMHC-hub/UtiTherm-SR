"""Step 11D: leakage-checked, sequence-held-out feasibility of cheap ROI selection.

Run from root:
  python scripts/audit_bicubic_first_predictability_step11d.py
  python scripts/audit_bicubic_first_predictability_step11d.py --config configs/bicubic_first_predictability_step11d.yaml

Inputs: existing Step11A ROI CSV + original low-resolution detector-TRAIN images.
No detector or SR inference, only cheap LR features and small CPU models.
Outcomes are ONE-ROI marginal utility SUM PROXIES, *not* combined ROI detections.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROTOCOL = 'STEP_11D_BICUBIC_FIRST_CHEAP_FEATURE_PREDICTABILITY_V1'
FEATURES = (
    'tile_mean','tile_std','tile_p10','tile_p90','tile_dynamic_range',
    'tile_bright_fraction','tile_dark_fraction','tile_entropy',
    'tile_grad_mean','tile_grad_std','tile_edge_fraction',
    'tile_laplacian_mean','tile_laplacian_std',
    'global_mean','global_std','relative_brightness','relative_grad',
    'center_x','center_y','distance_to_center',
)
REQUIRED_COLUMNS = (
    'image','sequence','tile_index','tile_x0_hr','tile_y0_hr',
    'tile_x1_hr','tile_y1_hr','baseline_kind','processed_lr_pixels',
    'baseline_small_tp','baseline_all_tp','baseline_fp',
    'candidate_small_tp','candidate_all_tp','candidate_fp',
    'delta_small_tp','delta_all_tp','delta_fp','utility_score','positive_label',
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sequence_id(filename: str) -> str:
    return filename.split('-frame-', 1)[0] if '-frame-' in filename else filename.rsplit('.', 1)[0]


def validated_rows(csv_path: Path, expected_tiles: int, fp_penalty: float) -> tuple[list[dict], dict]:
    with csv_path.open('r', newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or set(REQUIRED_COLUMNS) - set(reader.fieldnames):
            raise ValueError('Step 11A ROI CSV missing required columns')
        rows = list(reader)
    by_image: dict[str,list[dict]] = defaultdict(list)
    seen = set()
    for row in rows:
        name, seq = row['image'], row['sequence']
        if not name.endswith('.png') or seq != sequence_id(name):
            raise ValueError(f'Invalid image/sequence: {name}')
        tile = int(row['tile_index'])
        key = (name, tile)
        if key in seen or tile < 0 or tile >= expected_tiles:
            raise ValueError(f'Duplicate or invalid tile: {key}')
        seen.add(key)
        if row['baseline_kind'] != 'full_bicubic_640':
            raise ValueError('This audit requires full Bicubic as baseline')
        if int(row['processed_lr_pixels']) != 8400:
            raise ValueError('Expected single ROI halo10 processed pixel count 8400')
        dtp = int(row['candidate_small_tp']) - int(row['baseline_small_tp'])
        dall = int(row['candidate_all_tp']) - int(row['baseline_all_tp'])
        dfp = int(row['candidate_fp']) - int(row['baseline_fp'])
        u = dtp - fp_penalty * dfp
        if (int(row['delta_small_tp']) != dtp or int(row['delta_all_tp']) != dall or
            int(row['delta_fp']) != dfp or not math.isfinite(float(row['utility_score'])) or
            abs(float(row['utility_score']) - u) > 1e-5 or
            int(row['positive_label']) != int(u > 0)):
            raise ValueError(f'Utility mismatch in {key}')
        # Frozen 4x4 HR grid, row-major, 160x128 HR tiles.
        col, rr = tile % 4, tile // 4
        expected_coords = (160*col,128*rr,160*(col+1),128*(rr+1))
        got_coords = tuple(int(row[k]) for k in ('tile_x0_hr','tile_y0_hr','tile_x1_hr','tile_y1_hr'))
        if got_coords != expected_coords:
            raise ValueError(f'Tile coords wrong in {key}: {got_coords}')
        row['utility_float'] = float(row['utility_score'])
        row['positive_int'] = int(row['positive_label'])
        row['tile_int'] = tile
        by_image[name].append(row)
    for name, group in by_image.items():
        if (len(group) != expected_tiles or
            {r['tile_int'] for r in group} != set(range(expected_tiles)) or
            len({(r['baseline_small_tp'],r['baseline_all_tp'],r['baseline_fp']) for r in group}) != 1):
            raise ValueError(f'Incomplete image or inconsistent baseline: {name}')
    rows.sort(key=lambda r:(r['image'],r['tile_int']))
    return rows, by_image


def entropy_uint8(tile: np.ndarray) -> float:
    counts = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
    p = counts[counts > 0] / float(tile.size)
    return float(-(p * np.log2(p)).sum())


def extract_frame_features(gray: np.ndarray, tile_coords: list[tuple[int,int,int,int]]) -> np.ndarray:
    """Only LR pixels/coordinates; never uses detector predictions or labels."""
    if gray.shape != (256,320) or gray.dtype != np.uint8:
        raise ValueError('Required uint8 grayscale LR shape (256,320)')
    im = gray.astype(np.float32)
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx,gy)
    lap = np.abs(cv2.Laplacian(im,cv2.CV_32F,ksize=3))
    gl_mean, gl_std, grad_global = float(im.mean()),float(im.std()),float(gradient.mean())
    features = []
    for x0,y0,x1,y1 in tile_coords:
        if not (0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 512) or any(v % 2 for v in (x0,y0,x1,y1)):
            raise ValueError('Invalid HR tile rectangle')
        t = im[y0//2:y1//2,x0//2:x1//2]
        g = gradient[y0//2:y1//2,x0//2:x1//2]
        l = lap[y0//2:y1//2,x0//2:x1//2]
        tm, ts, gm = float(t.mean()),float(t.std()),float(g.mean())
        p10,p90 = np.percentile(t,[10,90])
        cx,cy = (x0+x1)/(2*640),(y0+y1)/(2*512)
        feat = [tm,ts,float(p10),float(p90),float(p90-p10),
                float(np.mean(t>192)),float(np.mean(t<64)),
                entropy_uint8(t.astype(np.uint8)),gm,float(g.std()),float(np.mean(g>40)),
                float(l.mean()),float(l.std()),
                gl_mean,gl_std,tm-gl_mean,gm-grad_global,
                cx,cy,float(math.hypot(cx-.5,cy-.5))]
        features.append(feat)
    mat = np.array(features, dtype=np.float64)
    if mat.shape != (len(tile_coords),len(FEATURES)) or not np.isfinite(mat).all():
        raise ValueError('Invalid generated features')
    return mat


def split_groups(seq_ids: list[str], seed: int, val_frac: float, test_frac: float) -> dict[str,list[str]]:
    if not 0 < val_frac < 0.4 or not 0 < test_frac < 0.4 or val_frac+test_frac >= .6:
        raise ValueError('Fractions must be reasonable and non-overlapping')
    seq_ids = sorted(set(seq_ids))
    if len(seq_ids) < 6:
        raise ValueError('Need >=6 sequences for group splitting')
    rng = np.random.default_rng(seed)
    perm = [seq_ids[int(i)] for i in rng.permutation(len(seq_ids))]
    ntest = max(1,int(math.ceil(test_frac*len(perm))))
    nval = max(1,int(math.ceil(val_frac*len(perm))))
    return {'train':sorted(perm[ntest+nval:]),'val':sorted(perm[ntest:ntest+nval]),'test':sorted(perm[:ntest])}


def choose_indices(scores: np.ndarray, k: int, threshold: float | None = None) -> list[int]:
    """Top-k with stable tie-breaking by tile index; optionally abstain."""
    score = np.asarray(scores, dtype=float).reshape(-1)
    if not np.isfinite(score).all() or k < 0 or k > len(score):
        raise ValueError('Invalid ranking scores or tile budget')
    ranked = sorted(range(len(score)),key=lambda i:(-float(score[i]),i))
    if threshold is not None:
        ranked = [i for i in ranked if score[i] > threshold]
    return ranked[:k]


def score_selections(images: list[str], values: dict[str,np.ndarray],
                     ranks: dict[str,np.ndarray], k: int, threshold: float | None = None,
                     positivity: dict[str,np.ndarray] | None = None) -> dict:
    per_image = {}
    selected_total, positive_selected, negative_selected = 0,0,0
    for image in images:
        choice = choose_indices(ranks[image],k,threshold)
        gains = values[image]
        per_image[image] = float(sum(float(gains[i]) for i in choice))
        selected_total += len(choice)
        positive_selected += sum(gains[i]>0 for i in choice)
        negative_selected += sum(gains[i]<0 for i in choice)
    mean = float(np.mean(list(per_image.values()))) if images else float('nan')
    return {'mean_marginal_utility_per_image':mean,'selected_regions':selected_total,
            'positive_selected':int(positive_selected),'negative_selected':int(negative_selected),
            'images_abstaining':sum(not choose_indices(ranks[i],k,threshold) for i in images),
            'per_image':per_image}


def exact_random_expectation(images: list[str], values: dict[str,np.ndarray], k: int) -> dict:
    # Exact expectation over uniformly sampled k distinct tiles, not one random draw.
    per_image = {im:float(k*np.mean(values[im])) for im in images}
    return {'mean_marginal_utility_per_image':float(np.mean(list(per_image.values()))),'per_image':per_image}


def cluster_bootstrap_difference(a: dict[str,float], b: dict[str,float],
                                 seq_for_image: dict[str,str], seed: int,
                                 iterations: int) -> tuple[float,float]:
    if set(a) != set(b) or not a or iterations < 100:
        raise ValueError('Need matched image outcomes and bootstrap iterations >=100')
    groups: dict[str,list[float]] = defaultdict(list)
    for im in sorted(a):
        groups[seq_for_image[im]].append(a[im]-b[im])
    groups_np = [np.asarray(groups[s]) for s in sorted(groups)]
    rng = np.random.default_rng(seed)
    stats = np.zeros(iterations)
    for i in range(iterations):
        choices = rng.integers(0,len(groups_np),size=len(groups_np))
        stats[i] = sum(groups_np[j].sum() for j in choices)/sum(len(groups_np[j]) for j in choices)
    return tuple(float(v) for v in np.quantile(stats,[.025,.975]))


def safe_ap(y: np.ndarray, scores: np.ndarray) -> float | None:
    if len(y) != len(scores) or len(set(y.tolist())) < 2:
        return None
    return float(average_precision_score(y,scores))


def mean_dict_values(a:dict[str,float]) -> float:
    return float(np.mean(list(a.values())))


def save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def run(args: argparse.Namespace) -> dict:
    cfgfile = args.config
    cfg = yaml.safe_load(cfgfile.read_text(encoding='utf-8'))
    if not isinstance(cfg,dict):
        raise ValueError('YAML must be a mapping')
    if cfg.get('scoring') != 'marginal_one_tile_utility_sum_proxy_ONLY':
        raise ValueError('Incorrect protocol label')
    csvfile = Path(cfg['source_csv'])
    source_summary_file = Path(cfg['source_summary'])
    source_summary = json.loads(source_summary_file.read_text(encoding='utf-8'))
    if (source_summary['protocol'] != 'STEP_11A_INCREMENTAL_LARGE_FSRCNN_OVER_BICUBIC_SINGLE_ROI_V1' or
        source_summary['roi_csv_sha256'] != sha256(csvfile) or
        source_summary['baseline'] != 'full_bicubic_640'):
        raise ValueError('Source summary/checksum/protocol mismatch')
    rows, by_image = validated_rows(csvfile,int(cfg['expected_tiles_per_image']),float(cfg['fp_penalty']))
    images = sorted(by_image)
    sequences = sorted({r['sequence'] for r in rows})
    if (len(images) != int(cfg['expected_images']) or len(sequences) != int(cfg['expected_sequences']) or
        len(rows) != int(cfg['expected_interventions']) or len(images)*16 != len(rows) or
        sum(r['positive_int'] for r in rows) != int(source_summary['positive_utility'])):
        raise ValueError('Counts or positive label totals differ from locked inputs')
    if set(images) != set(source_summary['source_provenance']['selected_images']):
        raise ValueError('CSV images differ from source provenance')
    budget = int(cfg['budget_tiles'])
    if budget != 4:
        raise ValueError('Step 11D requires frozen four-tile budget')
    if (int(cfg['image_width_lr']),int(cfg['image_height_lr'])) != (320,256):
        raise ValueError('LR dimensions do not match source protocol')
    cv2.setNumThreads(1)
    f_by_im,y_by_im,p_by_im = {},{},{ }
    seq_for_im = {}
    path_root = Path(cfg['train_lr'])
    for ix,image in enumerate(images,1):
        img_file=path_root/image
        gray = cv2.imread(str(img_file),cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f'Missing LR train image: {img_file}')
        if gray.shape != (256,320) or gray.dtype != np.uint8:
            raise ValueError(f'Wrong LR image dimensions/format: {image}')
        rr = sorted(by_image[image],key=lambda r:r['tile_int'])
        rects = [tuple(int(r[k]) for k in ('tile_x0_hr','tile_y0_hr','tile_x1_hr','tile_y1_hr')) for r in rr]
        f_by_im[image] = extract_frame_features(gray,rects)
        y_by_im[image] = np.array([r['utility_float'] for r in rr],dtype=float)
        p_by_im[image] = np.array([r['positive_int'] for r in rr],dtype=int)
        seq_for_im[image] = rr[0]['sequence']
        if ix==1 or ix % 50==0 or ix==len(images):
            print(f'Features extracted {ix}/{len(images)} frames',flush=True)
    split = split_groups(sequences,int(cfg['seed']),float(cfg['val_sequence_fraction']),float(cfg['test_sequence_fraction']))
    sets = {name:[im for im in images if seq_for_im[im] in ss] for name,ss in split.items()}
    for k in ('train','val','test'):
        if not sets[k]:
            raise ValueError('Empty split')
    x_train = np.vstack([f_by_im[im] for im in sets['train']])
    y_train = np.concatenate([y_by_im[im] for im in sets['train']])
    p_train = (y_train>0).astype(int)
    if len(set(p_train.tolist())) != 2:
        raise ValueError('No positive/negative training variation')
    reg=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=4,max_depth=10,
            max_features=.85,n_jobs=2,random_state=int(cfg['seed']))
    reg.fit(x_train,y_train)
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1200,class_weight='balanced',
            random_state=int(cfg['seed']),solver='lbfgs'))
    clf.fit(x_train,p_train)
    policies = {
        'edge_top4': {im: f_by_im[im][:,FEATURES.index('tile_grad_mean')] for im in images},
        'contrast_top4': {im: f_by_im[im][:,FEATURES.index('tile_std')] for im in images},
        'extra_trees_top4': {im: reg.predict(f_by_im[im]) for im in images},
        'logistic_top4': {im: clf.predict_proba(f_by_im[im])[:,1] for im in images},
    }
    # Tune abstention on validation ONLY; strictly fixed grids (not test searched).
    grids = {'extra_trees':[-.2,-.1,-.05,0,.025,.05,.1,.2,.4,1],
             'logistic':[.1,.2,.3,.4,.5,.6,.7,.8,.9,1]}
    thresholds={}
    for model_name,grid in grids.items():
        ranks=policies[f'{model_name}_top4']
        candidates=[(score_selections(sets['val'],y_by_im,ranks,budget,t)['mean_marginal_utility_per_image'],t)
                    for t in grid]
        # Favor more conservative threshold in utility ties; avoid selective test use.
        best=max(candidates,key=lambda q:(q[0],q[1]))
        thresholds[model_name] = {'threshold':best[1],'validation_mean_marginal_utility_per_image':best[0],
                                  'grid':grid}
        policies[f'{model_name}_gated'] = ranks
    report_splits = {}
    for label,subset in sets.items():
        methods={}
        for policy_name, ranks in policies.items():
            gate_name=policy_name.rsplit('_gated',1)[0] if policy_name.endswith('_gated') else None
            threshold=thresholds[gate_name]['threshold'] if gate_name else None
            scored=score_selections(subset,y_by_im,ranks,budget,threshold)
            all_targets=np.concatenate([p_by_im[im] for im in subset])
            all_ranks=np.concatenate([ranks[im] for im in subset])
            scored['average_precision_positive_roi']=safe_ap(all_targets,all_ranks)
            methods[policy_name]=scored
        methods['random4_expectation']=exact_random_expectation(subset,y_by_im,budget)
        methods['first4']=score_selections(subset,y_by_im,{im:-np.arange(16,dtype=float) for im in subset},budget)
        methods['oracle_top4_hindsight']=score_selections(subset,y_by_im,{im:y_by_im[im] for im in subset},budget)
        methods['oracle_gated_hindsight']=score_selections(subset,y_by_im,{im:y_by_im[im] for im in subset},budget,0.0)
        methods['bicubic_no_neural_sr']={
            'mean_marginal_utility_per_image':0.0,
            'per_image':{im:0.0 for im in subset},'selected_regions':0,'images_abstaining':len(subset)}
        # Pairwise CIs only on test; val is used for threshold choice.
        tests={}
        if label=='test':
            for a,b in [('extra_trees_top4','random4_expectation'),
                         ('logistic_top4','random4_expectation'),
                         ('extra_trees_gated','bicubic_no_neural_sr'),
                         ('logistic_gated','bicubic_no_neural_sr'),
                         ('extra_trees_gated','random4_expectation'),
                         ('logistic_gated','extra_trees_gated')]:
                aa=methods[a]['per_image']; bb=methods[b]['per_image']
                diff={im:aa[im]-bb[im] for im in subset}
                ci=cluster_bootstrap_difference(aa,bb,seq_for_im,int(cfg['seed'])+31,
                                                  int(cfg['bootstrap_iterations']))
                tests[f'{a}__vs__{b}']={'difference_per_image':mean_dict_values(diff),
                        'sequence_bootstrap_95ci':list(ci),'num_sequences':len(split['test']),
                        'image_wins_ties_losses':[sum(d>0 for d in diff.values()),
                                                   sum(d==0 for d in diff.values()),sum(d<0 for d in diff.values())]}
        # Per image data is exported separately and not bloated into summary.
        report_splits[label]={
            'num_images':len(subset),'num_sequences':len(split[label]),
            'positive_tile_fraction':float(np.mean(np.concatenate([p_by_im[im] for im in subset]))),
            'methods':{k:{field:v for field,v in m.items() if field!='per_image'} for k,m in methods.items()},
            'comparisons':tests}
        if label=='test':
            print('TEST - one-ROI marginal-sum PROXY only (not combined detection):',flush=True)
            for name in ('bicubic_no_neural_sr','random4_expectation','edge_top4','contrast_top4',
                         'extra_trees_top4','logistic_top4','extra_trees_gated','logistic_gated',
                         'oracle_top4_hindsight','oracle_gated_hindsight'):
                m=methods[name]
                print(f'  {name:28s} {m["mean_marginal_utility_per_image"]:+.4f} / image',flush=True)
    out=Path(args.out) if args.out else Path(cfg['out'])
    if out.exists() and not args.overwrite:
        raise FileExistsError(f'Output exists: {out}; --overwrite required to replace results')
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/'sequence_split.json',{
        'seed':cfg['seed'],'sequence_ids':split,
        'images_by_split':sets,
        'csv_sha256':sha256(csvfile)})
    with (out/'test_per_image.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file)
        names=('bicubic_no_neural_sr','random4_expectation','first4','edge_top4','contrast_top4',
               'extra_trees_top4','logistic_top4','extra_trees_gated','logistic_gated',
               'oracle_top4_hindsight','oracle_gated_hindsight')
        writer.writerow(['image','sequence',*names])
        for im in sets['test']:
            # reconstruct selections for test using same policies/thresholds.
            vals=[]
            for key in names:
                if key=='bicubic_no_neural_sr':vals.append(0)
                elif key=='random4_expectation':vals.append(float(budget*np.mean(y_by_im[im])))
                elif key.startswith('oracle_'):
                    vals.append(sum(y_by_im[im][j] for j in choose_indices(y_by_im[im],budget,
                                                       0.0 if key=='oracle_gated_hindsight' else None)))
                elif key=='first4':vals.append(sum(y_by_im[im][:budget]))
                else:
                    keymodel=key.replace('_gated','_top4')
                    gate=key.split('_gated')[0] if key.endswith('_gated') else None
                    picks=choose_indices(policies[keymodel][im],budget,
                                         thresholds[gate]['threshold'] if gate else None)
                    vals.append(sum(y_by_im[im][j] for j in picks))
            writer.writerow([im,seq_for_im[im],*[round(float(v),6) for v in vals]])
    report={'task':'Step 11D: pre-SR low-cost feature predictability audit over Bicubic',
            'created_utc':datetime.now(timezone.utc).isoformat(),'protocol':PROTOCOL,
            'config_sha256':sha256(cfgfile),'source_csv_sha256':sha256(csvfile),
            'source_summary_sha256':sha256(source_summary_file),
            'num_images':len(images),'num_sequences':len(sequences),'num_tiles':len(rows),
            'features':list(FEATURES),
            'training_sources':'LR thermal pixels, tile coordinates and Step 11A utility labels ONLY',
            'train_val_test_sequence_counts':{name:len(ss) for name,ss in split.items()},
            'train_val_test_image_counts':{name:len(ss) for name,ss in sets.items()},
            'model_selection':'fixed configs; gating thresholds chosen ONLY on sequence-separated validation',
            'gating_thresholds':thresholds,
            'results':report_splits,
            'limitations':[
                'All labels from detector-TRAIN images; fixed detector is in-sample. Exploratory.',
                'Source frames used in earlier development; held-out utility-model TEST is not an untouched independent benchmark.',
                'Single-ROI interventions cannot be added to infer actual four-ROI outcomes.',
                'Oracle rankings use hindsight labels and are not deployable or combined-ROI upper bounds.',
                'Exact random expectation is analytic, not a measured random four-region combined result.',
                'No neural-SR inference, no combined four-ROI evaluation, no standard detection mAP.',
                'Feature extraction + CPU predictor runtime NOT benchmarked on target RTX 5070.',
                'Only 5 held-out video groups: CIs are exploratory, not conclusive.',
            ]}
    save_json(out/'summary.json',report)
    print(f'DONE — {out / "summary.json"}; test sequences={len(split["test"])}',flush=True)
    return report


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path('configs/bicubic_first_predictability_step11d.yaml'))
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--overwrite',action='store_true')
    run(ap.parse_args())


if __name__=='__main__':
    main()
