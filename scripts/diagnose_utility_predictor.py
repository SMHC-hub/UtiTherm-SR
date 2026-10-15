"""Step 7B1: audit and diagnose fixed Step 7A FLIR development results.

Only reads existing CSV/JSON. Does not train, choose regions, or touch checkpoints.
Uses 17 sequence clusters rather than treating 200 correlated frames as independent.
This is development analysis, NOT an untouched final test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

MODES = (
    'linear_640', 'full_bicubic_640', 'full_fsrcnn_640', 'fixed_bicubic4',
    'random_bicubic4', 'predictor_bicubic4',
    'random_batched_fsrcnn4', 'predictor_batched_fsrcnn4',
)
PAIRS = (
    ('predictor_bicubic4', 'random_bicubic4'),
    ('predictor_bicubic4', 'fixed_bicubic4'),
    ('predictor_bicubic4', 'linear_640'),
    ('predictor_bicubic4', 'full_bicubic_640'),
    ('predictor_batched_fsrcnn4', 'random_batched_fsrcnn4'),
    ('predictor_batched_fsrcnn4', 'full_fsrcnn_640'),
)
FP_COSTS = (0.0, 0.25, 0.5, 1.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path, required: set[str]) -> list[dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or (required - set(reader.fieldnames)):
            raise ValueError(f'Missing columns in {path}: {sorted(required - set(reader.fieldnames or []))}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'No rows in {path}')
    return rows


def sequence_id(image: str) -> str:
    if Path(image).name != image or not image.endswith('.png') or '-frame-' not in image:
        raise ValueError(f'Unrecognized FLIR image basename: {image}')
    return image.split('-frame-', 1)[0]


def audit_detection(rows: list[dict], summary: dict, manifest: dict) -> dict[str, dict[str, dict]]:
    """Validate all pairwise policies, matching GT, frozen split, and report totals."""
    if summary.get('protocol') != 'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST':
        raise ValueError('Refusing report with unexpected protocol')
    if summary.get('split_manifest_sha256') != sha256(Path(manifest['_path'])):
        raise ValueError('Step 7A report does not match supplied frozen manifest')
    frozen = manifest.get('images', [])
    if not frozen or len(frozen) != len(set(frozen)):
        raise ValueError('Invalid frozen image list')
    by_image: dict[str, dict[str, dict]] = defaultdict(dict)
    for raw in rows:
        im, m, seq = raw['image'], raw['method'], raw['sequence']
        if seq != sequence_id(im):
            raise ValueError(f'Sequence mismatch: {im}')
        if m not in MODES or m in by_image[im]:
            raise ValueError(f'Duplicate/unrecognized method for image {im}: {m}')
        row = dict(raw)
        for col in ('small_gt', 'small_tp', 'all_tp', 'fp'):
            n = int(raw[col])
            if n < 0:
                raise ValueError(f'Negative {col} for {im}')
            row[col] = n
        if row['small_tp'] > row['small_gt'] or row['small_tp'] > row['all_tp']:
            raise ValueError(f'Impossible TP totals: {im} {m}')
        by_image[im][m] = row
    if set(by_image) != set(frozen):
        raise ValueError('Evaluated image names do not match frozen manifest')
    for im, ms in by_image.items():
        if set(ms) != set(MODES):
            raise ValueError(f'Missing method for {im}: {sorted(set(MODES)-set(ms))}')
        if len({v['small_gt'] for v in ms.values()}) != 1:
            raise ValueError(f'Ground-truth counts differ by method: {im}')
    report = summary['metrics']
    if report['images'] != len(by_image) or report['unique_sequences'] != len({sequence_id(im) for im in by_image}):
        raise ValueError('Report image/sequence counts inconsistent')
    for m in MODES:
        for col in ('small_gt', 'small_tp', 'all_tp', 'fp'):
            computed = sum(ms[m][col] for ms in by_image.values())
            if computed != int(report['methods'][m][col]):
                raise ValueError(f'CSV/report mismatch: method={m}, column={col}, {computed} != {report["methods"][m][col]}')
    return dict(by_image)


def paired_deltas(by_image: dict, a: str, b: str, fp_cost: float) -> list[dict]:
    result = []
    for im in sorted(by_image):
        x, y = by_image[im][a], by_image[im][b]
        dtp = x['small_tp'] - y['small_tp']
        dfp = x['fp'] - y['fp']
        result.append({'image': im, 'sequence': sequence_id(im), 'delta_small_tp': dtp,
                       'delta_fp': dfp, 'delta_utility': dtp - fp_cost * dfp})
    return result


def sequence_bootstrap_ci(rows: list[dict], field: str, seed: int, resamples: int = 2000) -> dict:
    """Equal weight per video sequence; resample clusters, not frames."""
    if resamples < 100:
        raise ValueError('At least 100 resamples needed')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['sequence']].append(float(row[field]))
    if len(grouped) < 2:
        raise ValueError('At least two video sequences needed')
    seq_mean = [mean(vals) for _, vals in sorted(grouped.items())]
    n = len(seq_mean)
    rng = random.Random(seed)
    samples = sorted(mean(seq_mean[rng.randrange(n)] for _ in range(n)) for _ in range(resamples))
    lo = samples[int(0.025 * (resamples-1))]
    hi = samples[int(0.975 * (resamples-1))]
    return {'mean_per_sequence_equal_weight': round(mean(seq_mean), 6),
            'sequence_bootstrap_95ci': [round(lo, 6), round(hi, 6)],
            'sequence_count': n,
            'note': 'Exploratory interval; limited to filename-derived video groups and evaluated development set.'}


def summarise_pair(rows: list[dict], name_a: str, name_b: str, fp_cost: float, seed: int,
                   bootstraps: int) -> dict:
    tp = sum(r['delta_small_tp'] for r in rows)
    fp = sum(r['delta_fp'] for r in rows)
    n = len(rows)
    cluster = sequence_bootstrap_ci(rows, 'delta_utility', seed, bootstraps)
    return {'method': name_a, 'comparator': name_b, 'fp_penalty': fp_cost,
            'additional_small_tp': tp, 'additional_fp': fp,
            'total_utility_difference': round(tp - fp_cost*fp, 6),
            'utility_difference_per_image': round((tp - fp_cost*fp)/n, 6),
            'images_better_utility': sum(r['delta_utility'] > 0 for r in rows),
            'images_worse_utility': sum(r['delta_utility'] < 0 for r in rows),
            'images_tied_utility': sum(r['delta_utility'] == 0 for r in rows),
            **cluster}


def label_profile(rows: list[dict]) -> dict:
    """Step 5A label mix is training-domain context, not 200-image test labels."""
    required = {'image', 'sequence', 'tile_index', 'delta_small_tp', 'delta_fp', 'utility_score', 'positive_label'}
    if not rows or not required <= set(rows[0]):
        raise ValueError('Step 5A CSV missing fields')
    unique = set()
    seqs = set()
    counts = Counter()
    n_imgs = Counter()
    for row in rows:
        im = row['image']
        if row['sequence'] != sequence_id(im):
            raise ValueError(f'Invalid ROI sequence: {im}')
        key = (im, int(row['tile_index']))
        if key in unique or key[1] not in range(16):
            raise ValueError(f'Duplicate/invalid utility ROI: {key}')
        unique.add(key)
        seqs.add(row['sequence'])
        n_imgs[im] += 1
        d_tp, d_fp = int(row['delta_small_tp']), int(row['delta_fp'])
        utility = float(row['utility_score'])
        if not math.isfinite(utility) or abs(utility - (d_tp - 0.25*d_fp)) > 1e-4:
            raise ValueError(f'Utility formula discrepancy in Step 5A CSV: {im}')
        if int(row['positive_label']) != int(utility > 0):
            raise ValueError(f'Incorrect positive utility label: {im}')
        counts['small_gain'] += d_tp > 0
        counts['small_loss'] += d_tp < 0
        counts['fp_reduction'] += d_fp < 0
        counts['fp_increase'] += d_fp > 0
        counts['positive_utility'] += utility > 0
        counts['positive_from_fp_only'] += (utility > 0 and d_tp == 0 and d_fp < 0)
        counts['positive_with_small_gain'] += (utility > 0 and d_tp > 0)
        counts['positive_other'] += (utility > 0 and d_tp <= 0 and not (d_tp == 0 and d_fp < 0))
    if any(v != 16 for v in n_imgs.values()):
        raise ValueError('Missing one or more ROI records per training image')
    n = len(rows)
    return {'source': 'Step 5A TRAIN-domain intervention labels (NOT the Step 7A 200-frame evaluation)',
            'images': len(n_imgs), 'sequences': len(seqs), 'rows': n,
            'counts': dict(counts),
            'rates_percent': {key: round(counts[key]/n*100, 2) for key in sorted(counts)}}


def write_csv(path: Path, data: list[dict]):
    if not data:
        raise ValueError('Cannot write empty CSV')
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def run(report: Path, manifest_path: Path, roi_csv: Path, out: Path,
        seed: int = 42, bootstraps: int = 2000) -> dict:
    if out.exists():
        raise FileExistsError(f'Experiment already exists: {out}')
    summary = json.loads((report/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['_path'] = str(manifest_path)
    dets = read_csv(report/'detection_per_image.csv', {'image','sequence','method','small_gt','small_tp','all_tp','fp'})
    by_image = audit_detection(dets, summary, manifest)
    seq_rows = []
    pair_rows = []
    for i, (a,b) in enumerate(PAIRS):
        per_image = paired_deltas(by_image, a,b, .25)
        pair_rows.append(summarise_pair(per_image,a,b,.25,seed+i,bootstraps))
        grouping = defaultdict(list)
        for row in per_image:
            grouping[row['sequence']].append(row)
        for seq, rs in sorted(grouping.items()):
            seq_rows.append({'method': a,'comparator':b,'sequence':seq, 'images':len(rs),
                'delta_small_tp':sum(r['delta_small_tp'] for r in rs),
                'delta_fp':sum(r['delta_fp'] for r in rs),
                'utility_delta_total':round(sum(r['delta_utility'] for r in rs),4),
                'utility_delta_per_frame':round(mean(r['delta_utility'] for r in rs),4)})
    # Changing FP penalty is an analysis only, not tuning or a new evaluation run.
    sensitivity=[]
    for a,b in PAIRS:
        for cost in FP_COSTS:
            rows = paired_deltas(by_image,a,b,cost)
            sensitivity.append({'method':a, 'comparator':b, 'fp_penalty':cost,
                 'total_utility_difference':round(sum(r['delta_utility'] for r in rows),4)})
    training_rows = read_csv(roi_csv, {'image','sequence','tile_index','delta_small_tp','delta_fp','utility_score','positive_label'})
    labels = label_profile(training_rows)
    # Create the report only after every integrity and consistency check passes.
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out/'pairwise_comparisons.csv', pair_rows)
    write_csv(out/'sequence_differences.csv', seq_rows)
    write_csv(out/'fp_penalty_sensitivity.csv', sensitivity)
    # Examples for qualitative investigation; cannot pinpoint which tile without selection logs.
    image_deltas = paired_deltas(by_image,'predictor_bicubic4','random_bicubic4',.25)
    image_deltas.sort(key=lambda x:(x['delta_utility'],x['image']))
    write_csv(out/'worst_images_vs_random.csv', image_deltas[:min(30,len(image_deltas))])
    diagnostic = {
        'task':'UtiTherm-SR Step 7B1: utility predictor diagnosis',
        'protocol':'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST',
        'report_sha256':sha256(report/'summary.json'),
        'manifest_sha256':sha256(manifest_path),
        'roi_training_csv_sha256':sha256(roi_csv),
        'images':len(by_image),
        'sequences':len({sequence_id(k) for k in by_image}),
        'fp_penalty':0.25,
        'pairwise':pair_rows,
        'training_label_profile':labels,
        'limitations':[
            'DEVELOPMENT dataset: FLIR VAL was used during prior baseline design; NOT final held-out test.',
            'Sequence-bootstrap intervals represent 17 filename-derived video groups, not independent physical scenes.',
            'Paired comparisons use one fixed random tile-selection seed; multiple seeds are still needed.',
            'ROI labels are detector-TRAIN-domain and may be optimistic; no out-of-fold label generation yet.',
            'The 7A CSV contains only per-image aggregate TP/FP, NOT chosen tiles, scores or object classes. Root causes cannot be identified from this report alone.',
            'Confidence=.25 and IoU=.5: diagnostic small-object TP and FP, NOT official AP/mAP.',
            'FP penalty sensitivity is exploratory; do NOT select a penalty using final test performance.',
        ]
    }
    (out/'diagnosis.json').write_text(json.dumps(diagnostic,indent=2)+'\n', encoding='utf-8')
    return diagnostic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,default=Path('reports/development/eval_200_v1'))
    p.add_argument('--manifest',type=Path,default=Path('reports/development/split_200_v2/manifest.json'))
    p.add_argument('--roi-csv',type=Path,default=Path('reports/utility/probe_train_100_v1/roi_utility.csv'))
    p.add_argument('--out',type=Path,default=Path('reports/utility/step7b1_diagnosis_v1'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bootstraps',type=int,default=2000)
    args = p.parse_args()
    result=run(args.report,args.manifest,args.roi_csv,args.out,args.seed,args.bootstraps)
    print(f'Step 7B1: validated {result["images"]} frames across {result["sequences"]} sequence groups')
    print('Paired diagnostic utility: ΔsmallTP - 0.25 × Δfalse-positives')
    for x in result['pairwise']:
        print(f'  {x["method"]:30} vs {x["comparator"]:25} '
              f'ΔTP={x["additional_small_tp"]:+4} ΔFP={x["additional_fp"]:+4} '
              f'Δutility={x["total_utility_difference"]:+8.2f} '
              f'sequence CI={x["sequence_bootstrap_95ci"]}')
    print(f'DONE — {args.out/"diagnosis.json"}; {args.out/"sequence_differences.csv"}')


if __name__ == '__main__':
    main()
