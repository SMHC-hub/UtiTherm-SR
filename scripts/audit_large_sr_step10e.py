"""Step 10E — read-only per-image/sequence and timing audit of Step 10D.

Run from the project root: python scripts/audit_large_sr_step10e.py
No torch/CUDA, image inference, dataset alteration, or checkpoint writes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, mean

import yaml

DETECTION_FIELDS = ('image', 'sequence', 'method', 'small_tp', 'all_tp', 'fp', 'small_gt')
TIMING_FIELDS = ('image', 'sequence', 'repeat', 'method', 'selector_ms', 'sr_ms',
                 'detector_ms', 'total_ms', 'computed_pixels', 'peak_torch_mib')
TIMING_NUMS = ('selector_ms', 'sr_ms', 'detector_ms', 'total_ms', 'computed_pixels', 'peak_torch_mib')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, fields: tuple[str, ...]) -> list[dict]:
    with path.open('r', newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not set(fields).issubset(reader.fieldnames):
            raise ValueError(f'{path}: missing columns {set(fields) - set(reader.fieldnames or ())}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty CSV')
    return rows


def as_nonnegative_int(text: str, label: str) -> int:
    try:
        number = int(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{label}: invalid integer {text!r}') from exc
    if number < 0:
        raise ValueError(f'{label}: negative integer {number}')
    return number


def as_finite_nonnegative_float(text: str, label: str) -> float:
    try:
        number = float(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{label}: invalid number {text!r}') from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f'{label}: invalid nonnegative finite number {text!r}')
    return number


def load_detections(path: Path, expected_images: int = 0, expected_sequences: int = 0):
    result = {}
    image_metadata = {}
    methods = set()
    for row in read_csv(path, DETECTION_FIELDS):
        image, sequence, method = (row[col].strip() for col in ('image', 'sequence', 'method'))
        if not image or not sequence or not method:
            raise ValueError('Detection row missing image, sequence, or method')
        key = (image, method)
        if key in result:
            raise ValueError(f'Duplicate detection record: {key}')
        nums = {name: as_nonnegative_int(row[name], f'{key} {name}')
                for name in ('small_tp', 'all_tp', 'fp', 'small_gt')}
        if nums['small_tp'] > nums['small_gt'] or nums['small_tp'] > nums['all_tp']:
            raise ValueError(f'Impossible TP counts: {key}')
        meta = (sequence, nums['small_gt'])
        if image in image_metadata and image_metadata[image] != meta:
            raise ValueError(f'GT or sequence mismatch between methods: {image}')
        image_metadata[image] = meta
        result[key] = {'image': image, 'sequence': sequence, 'method': method, **nums}
        methods.add(method)
    images = sorted(image_metadata)
    if expected_images and len(images) != expected_images:
        raise ValueError(f'Expected {expected_images} images, got {len(images)}')
    nseq = len({x[0] for x in image_metadata.values()})
    if expected_sequences and nseq != expected_sequences:
        raise ValueError(f'Expected {expected_sequences} sequences, got {nseq}')
    for image in images:
        present = {m for i, m in result if i == image}
        if present != methods:
            raise ValueError(f'Method records missing for image {image}: {sorted(methods - present)}')
    return result, images, sorted(methods), nseq


def load_timings(path: Path, images: list[str], methods: list[str], detections: dict,
                 expected_repeats: int = 0):
    grouped = defaultdict(list)
    repeats_seen = set()
    image_set, method_set = set(images), set(methods)
    for row in read_csv(path, TIMING_FIELDS):
        image, sequence, method = (row[k].strip() for k in ('image', 'sequence', 'method'))
        if image not in image_set or method not in method_set:
            raise ValueError(f'Timing has unknown image/method: {(image, method)}')
        if sequence != detections[(image, method)]['sequence']:
            raise ValueError(f'Timing sequence differs from detection: {(image, method)}')
        repeat = as_nonnegative_int(row['repeat'], f'{image}/{method} repeat')
        unique = (image, method, repeat)
        if unique in repeats_seen:
            raise ValueError(f'Duplicate timing trial {unique}')
        repeats_seen.add(unique)
        nums = {field: as_finite_nonnegative_float(row[field], f'{unique} {field}')
                for field in TIMING_NUMS}
        grouped[(image, method)].append((repeat, nums))
    per_image_medians = {}
    for image in images:
        for method in methods:
            key = (image, method)
            trials = grouped.get(key, [])
            if not trials:
                raise ValueError(f'Missing timing trials for {key}')
            if expected_repeats and len(trials) != expected_repeats:
                raise ValueError(f'{key}: expected {expected_repeats} trials, got {len(trials)}')
            rnums = sorted(repeat for repeat, _ in trials)
            if rnums != list(range(len(trials))):
                raise ValueError(f'{key}: repeats must be consecutive from 0, got {rnums}')
            per_image_medians[key] = {
                name: float(median(values[name] for _, values in trials))
                for name in TIMING_NUMS
            }
    return per_image_medians


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values or not 0 <= probability <= 1:
        raise ValueError('Percentile needs sorted nonempty list and p in [0,1]')
    index = (len(sorted_values) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index - lo)


def grouped_bootstrap_ci(per_image: list[dict], seed: int, n_boot: int,
                         field: str = 'delta_utility', alpha: float = 0.05) -> list[float]:
    """Resample video sequences, weighted by frames, NOT individual correlated frames."""
    if n_boot < 100 or not 0 < alpha < 1:
        raise ValueError('Bootstrap needs >=100 draws and alpha between 0 and 1')
    seq = defaultdict(lambda: [0.0, 0])
    for row in per_image:
        seq[row['sequence']][0] += row[field]
        seq[row['sequence']][1] += 1
    blocks = [seq[k] for k in sorted(seq)]
    if len(blocks) < 2:
        raise ValueError('Bootstrap needs at least two sequence groups')
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        total = 0.0
        count = 0
        for _ in blocks:
            numerator, denominator = blocks[rng.randrange(len(blocks))]
            total += numerator
            count += denominator
        values.append(total / count)
    values.sort()
    return [round(percentile(values, alpha / 2), 6),
            round(percentile(values, 1 - alpha / 2), 6)]


def write_csv(path: Path, records: list[dict], fieldnames: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def pairwise_rows(det: dict, times: dict, images: list[str], first: str, second: str,
                  fp_penalty: float) -> list[dict]:
    records = []
    for image in images:
        a, b = det[(image, first)], det[(image, second)]
        d_tp = a['small_tp'] - b['small_tp']
        d_fp = a['fp'] - b['fp']
        d_u = d_tp - fp_penalty * d_fp
        timing_diff = times[(image, first)]['total_ms'] - times[(image, second)]['total_ms']
        if d_tp > 0 and d_fp > 0:
            tradeoff = 'more_tp_more_fp'
        elif d_tp < 0 and d_fp < 0:
            tradeoff = 'fewer_tp_fewer_fp'
        elif d_tp >= 0 and d_fp <= 0 and (d_tp != 0 or d_fp != 0):
            tradeoff = 'tp_fp_pareto_better'
        elif d_tp <= 0 and d_fp >= 0 and (d_tp != 0 or d_fp != 0):
            tradeoff = 'tp_fp_pareto_worse'
        else:
            tradeoff = 'unchanged'
        records.append({'image': image, 'sequence': a['sequence'], 'method_a': first,
                        'method_b': second, 'small_gt': a['small_gt'],
                        'small_tp_a': a['small_tp'], 'small_tp_b': b['small_tp'],
                        'fp_a': a['fp'], 'fp_b': b['fp'],
                        'delta_small_tp': d_tp, 'delta_fp': d_fp,
                        'delta_utility': round(d_u, 6),
                        'delta_latency_ms': round(timing_diff, 6),
                        'tradeoff': tradeoff})
    return records


def analyze_pair(records: list[dict], first: str, second: str,
                 fp_weights: list[float], nboot: int, seed: int):
    seqgroups = defaultdict(list)
    for row in records:
        seqgroups[row['sequence']].append(row)
    per_sequence = []
    for seq in sorted(seqgroups):
        group = seqgroups[seq]
        utility = sum(row['delta_utility'] for row in group)
        per_sequence.append({'method_a': first, 'method_b': second, 'sequence': seq,
                             'num_images': len(group),
                             'delta_small_tp': sum(row['delta_small_tp'] for row in group),
                             'delta_fp': sum(row['delta_fp'] for row in group),
                             'delta_utility': round(utility, 6),
                             'mean_utility_per_image': round(utility / len(group), 6),
                             'mean_delta_latency_ms': round(mean(row['delta_latency_ms'] for row in group), 6)})
    wins = sum(row['delta_utility'] > 0 for row in records)
    ties = sum(row['delta_utility'] == 0 for row in records)
    losses = sum(row['delta_utility'] < 0 for row in records)
    seqwins = sum(row['delta_utility'] > 0 for row in per_sequence)
    seqties = sum(row['delta_utility'] == 0 for row in per_sequence)
    seqlosses = sum(row['delta_utility'] < 0 for row in per_sequence)
    tp = sum(row['delta_small_tp'] for row in records)
    fp = sum(row['delta_fp'] for row in records)
    utl = sum(row['delta_utility'] for row in records)
    lat = mean(row['delta_latency_ms'] for row in records)
    return {
        'method_a': first, 'method_b': second, 'num_images': len(records),
        'num_sequences': len(seqgroups), 'delta_small_tp': tp, 'delta_fp': fp,
        'delta_utility_total': round(utl, 6),
        'delta_utility_per_image': round(utl / len(records), 6),
        'utility_per_image_sequence_bootstrap_95ci': grouped_bootstrap_ci(records, seed, nboot, 'delta_utility'),
        'delta_mean_latency_ms': round(lat, 6),
        'latency_sequence_bootstrap_95ci_ms': grouped_bootstrap_ci(records, seed + 991, nboot, 'delta_latency_ms'),
        'image_utility_wins_ties_losses': {'wins': wins, 'ties': ties, 'losses': losses},
        'sequence_utility_wins_ties_losses': {'wins': seqwins, 'ties': seqties, 'losses': seqlosses},
        'image_tradeoff_counts': {kind: sum(row['tradeoff'] == kind for row in records)
                                  for kind in ('more_tp_more_fp', 'fewer_tp_fewer_fp',
                                               'tp_fp_pareto_better', 'tp_fp_pareto_worse', 'unchanged')},
        'utility_fp_penalty_sensitivity': [
            {'fp_penalty': penalty, 'delta_total_utility': round(tp - penalty * fp, 6)}
            for penalty in fp_weights
        ],
    }, per_sequence


def audit(cfg: dict) -> dict:
    fp_penalty = float(cfg.get('fp_penalty', 0.25))
    fp_weights = [float(x) for x in cfg.get('fp_penalty_sensitivity', [fp_penalty])]
    if not math.isfinite(fp_penalty) or fp_penalty < 0 or any(not math.isfinite(x) or x < 0 for x in fp_weights):
        raise ValueError('FP penalties must be finite and nonnegative')
    nboot = int(cfg.get('bootstrap_repetitions', 10000))
    seed = int(cfg.get('bootstrap_seed', 1510))
    det_path = Path(cfg['detection_csv'])
    timing_path = Path(cfg['timing_csv'])
    source_path = Path(cfg['source_summary'])
    out = Path(cfg['output_dir'])
    # Validate everything before creating output files.
    det, images, methods, nseq = load_detections(det_path, int(cfg.get('expected_images', 0)),
                                                 int(cfg.get('expected_sequences', 0)))
    times = load_timings(timing_path, images, methods, det, int(cfg.get('expected_repeats', 0)))
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if source.get('protocol') != 'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST':
        raise ValueError('Unexpected source protocol; refuse to treat this as Step 10D development data')
    if source.get('frozen_artifact_checks_verified') is not True:
        raise ValueError('Source report does not confirm frozen-artifact checks')
    src_results = source.get('results', {})
    if src_results.get('images') != len(images) or src_results.get('unique_sequences') != nseq:
        raise ValueError('Source summary image/sequence counts differ from CSV')
    if src_results.get('repeats_per_image') != int(cfg.get('expected_repeats', 0)):
        raise ValueError('Source summary repeats do not match audit configuration')
    aggregated = {}
    for method in methods:
        rows = [det[(image, method)] for image in images]
        trows = [times[(image, method)] for image in images]
        actual = {'small_gt': sum(r['small_gt'] for r in rows),
                  'small_tp': sum(r['small_tp'] for r in rows),
                  'all_tp': sum(r['all_tp'] for r in rows), 'fp': sum(r['fp'] for r in rows)}
        expected = src_results.get('methods', {}).get(method)
        if expected is None:
            raise ValueError(f'Source summary lacks method {method}')
        for metric, val in actual.items():
            if int(expected[metric]) != val:
                raise ValueError(f'{method}: source {metric}={expected[metric]} differs from CSV {val}')
        timings = {f'mean_{field}': round(mean(x[field] for x in trows), 6) for field in TIMING_NUMS}
        if abs(timings['mean_total_ms'] - float(expected['mean_of_image_median_total_ms'])) > 0.001:
            raise ValueError(f'{method}: source mean of per-image median latency differs from timing CSV')
        totals = sorted(t['total_ms'] for t in trows)
        aggregated[method] = {**actual, **timings,
                              'p50_per_image_median_total_ms': round(percentile(totals, 0.5), 6),
                              'p95_per_image_median_total_ms': round(percentile(totals, 0.95), 6),
                              'mean_unattributed_overhead_ms': round(mean(
                                  t['total_ms'] - t['selector_ms'] - t['sr_ms'] - t['detector_ms']
                                  for t in trows), 6)}
    pairs = cfg['pairs']
    if not pairs or any(not isinstance(p, list) or len(p) != 2 or p[0] == p[1] for p in pairs):
        raise ValueError('pairs must be nonempty two-element lists of different methods')
    for a, b in pairs:
        if a not in methods or b not in methods:
            raise ValueError(f'Unknown pair: {a}, {b}')
    per_image_all, per_seq_all, comparisons, worst = [], [], [], []
    for i, (a, b) in enumerate(pairs):
        rows = pairwise_rows(det, times, images, a, b, fp_penalty)
        stats, seqrows = analyze_pair(rows, a, b, fp_weights, nboot, seed + 11 * i)
        comparisons.append(stats)
        per_image_all.extend(rows)
        per_seq_all.extend(seqrows)
        worst.extend(sorted(rows, key=lambda r: (r['delta_utility'], r['image']))
                     [:int(cfg.get('worst_images_per_pair', 20))])
    result = {'task': 'Step 10E — read-only Step 10D paired detection and timing audit',
              'generated_utc': datetime.now(timezone.utc).isoformat(),
              'protocol': 'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST',
              'num_images': len(images), 'num_sequences': nseq,
              'fp_penalty': fp_penalty,
              'source_files': {str(p): sha256(p) for p in (det_path, timing_path, source_path)},
              'methods': aggregated, 'comparisons': comparisons,
              'limitations': [
                  'This is the repeatedly inspected FLIR validation development split, NOT an untouched final test.',
                  'The existing CSV has image-level TP/FP counts, not per-object matches or selected tile IDs.',
                  'A change in TP count does not identify the same individual objects; no causal ROI attribution is possible.',
                  'Four-ROI combined outcomes cannot be decomposed into single-ROI interactions from these files.',
                  'Bootstrap resamples video sequences, not correlated individual frames; 17 clusters are limited.',
                  'Timing excludes dataset disk IO/model checkpoint load and depends on hardware/runtime.',
                  'Utility weight 0.25 is a design choice; sensitivity values are descriptive, not tuned on test.',
                  'No official detection mAP is computed in this diagnostic audit.'
              ]}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'paired_per_image.csv', per_image_all,
              ['image', 'sequence', 'method_a', 'method_b', 'small_gt', 'small_tp_a', 'small_tp_b',
               'fp_a', 'fp_b', 'delta_small_tp', 'delta_fp', 'delta_utility', 'delta_latency_ms', 'tradeoff'])
    write_csv(out / 'paired_per_sequence.csv', per_seq_all,
              ['method_a', 'method_b', 'sequence', 'num_images', 'delta_small_tp', 'delta_fp',
               'delta_utility', 'mean_utility_per_image', 'mean_delta_latency_ms'])
    write_csv(out / 'worst_images.csv', worst,
              ['image', 'sequence', 'method_a', 'method_b', 'small_gt', 'small_tp_a', 'small_tp_b',
               'fp_a', 'fp_b', 'delta_small_tp', 'delta_fp', 'delta_utility', 'delta_latency_ms', 'tradeoff'])
    fieldnames = ['method', 'small_gt', 'small_tp', 'all_tp', 'fp'] + [
        f'mean_{name}' for name in TIMING_NUMS] + [
        'p50_per_image_median_total_ms', 'p95_per_image_median_total_ms',
        'mean_unattributed_overhead_ms']
    write_csv(out / 'timing_by_method.csv', [{'method': k, **v} for k, v in sorted(aggregated.items())], fieldnames)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/large_sr_error_audit_step10e.yaml'))
    parser.add_argument('--out', type=Path, help='Optional output directory override')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict):
        parser.error('Config must be a mapping')
    if args.out:
        cfg['output_dir'] = str(args.out)
    result = audit(cfg)
    print(f'STEP 10E DONE: {result["num_images"]} images, {result["num_sequences"]} sequences; read-only source')
    print('Paired comparisons (A minus B; sequence-bootstrap CI is for utility / image):')
    for comp in result['comparisons']:
        print(f'  {comp["method_a"]} vs {comp["method_b"]}: '
              f'ΔTP={comp["delta_small_tp"]:+d} ΔFP={comp["delta_fp"]:+d} '
              f'ΔU={comp["delta_utility_total"]:+.2f} '
              f'CI={comp["utility_per_image_sequence_bootstrap_95ci"]} '
              f'Δlatency={comp["delta_mean_latency_ms"]:+.3f}ms')
    print(f'SAVED — {cfg["output_dir"]}/summary.json')
    print('LIMITATION — development only; image counts cannot explain specific-object or tile interactions.')


if __name__ == '__main__':
    main()
