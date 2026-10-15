"""Step 10D: paired V2 vs V3 Large FSRCNN ROI detection + latency.

Runs on FROZEN Step 7A FLIR development frames (NOT a pristine final test set).
The existing Step 7A config/checkpoint SHA256 checks are preserved, even though
this experiment explicitly loads the trained Large SR model and V2/V3 selectors. Their
hashes and source provenance are recorded separately in the output report.

All method decisions use only LR pixels, frame name and frozen random seed;
GT labels are read *after* all methods have run on each image.

Usage (from repository root):
  python scripts/evaluate_v2_v3_large_step10d.py --limit 8 --repeats 3 \\
    --out reports/development/step10d_smoke_8_v1
  python scripts/evaluate_v2_v3_large_step10d.py --limit 0 --repeats 3 \\
    --out reports/development/step10d_v2_v3_200_v1
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
from prepare_development_split import (  # noqa: E402
    sequence_id, training_sequences, check_manifest, sha256,
)
from evaluate_selective_sr import (  # noqa: E402
    predictor_model, predict_tile_scores, choose_tiles,
)
from benchmark_selector_step8b import FastScorer  # noqa: E402
from probe_roi_utility import roi_grid, prediction_records, detection_counts  # noqa: E402
from evaluate_bicubic_baseline import ground_truth  # noqa: E402
from benchmark_selective_fsrcnn import load_sr_checkpoint  # noqa: E402
from utitherm.sr.roi_fsrcnn_batched import render_batched_fsrcnn  # noqa: E402
from utitherm.sr.roi_fsrcnn import render_fsrcnn  # noqa: E402

MODES = (
    'linear_640',
    'full_bicubic_640',
    'full_large_fsrcnn_640',
    'random_large_roi4_h16',
    'predictor_large_roi4_h16',
    'random_large_roi4_h10',
    'predictor_large_roi4_h10',
    'predictor_v3_large_roi4_h16',
    'predictor_v3_large_roi4_h10',
)
HR = (640, 512)
LR_SHAPE = (256, 320)


def validate_protocol(exp: dict, original: dict, manifest: dict) -> None:
    if exp.get('halos') != [10, 16] or exp.get('budget_tiles') != 4:
        raise ValueError('Step 10D requires frozen halo choices [10,16] and four tiles')
    if float(exp.get('false_positive_penalty', -1)) != 0.25:
        raise ValueError('Step 10D requires utility penalty=0.25')
    for key, expect in {
        'imgsz': 640, 'budget_tiles': 4, 'feather_hr_px': 8,
        'confidence': 0.25, 'nms_iou': 0.7,
        'false_positive_penalty': 0.25,
    }.items():
        if abs(float(original.get(key, -100)) - expect) > 1e-8:
            raise ValueError(f'Step 10D frozen evaluation setting changed: {key}')
    if str(original.get('device')) != '0':
        raise ValueError('Step 10D requires CUDA GPU 0')
    if int(manifest.get('selected_images', -1)) != int(exp.get('expected_images', -2)):
        raise ValueError('Frozen development selection size does not match experiment config')
    if int(manifest.get('unique_sequences', -1)) != int(exp.get('expected_sequences', -2)):
        raise ValueError('Frozen development sequence count does not match experiment config')


def assigned_tiles(method: str, name: str, scores: np.ndarray | None, seed: int) -> list[int]:
    if method.startswith('random_large_'):
        return choose_tiles('random4', name, 4, 16, seed)
    if method.startswith('predictor_large_') or method.startswith('predictor_v3_large_'):
        return choose_tiles('predictor4', name, 4, 16, seed, scores)
    if method not in MODES:
        raise ValueError(f'Unknown mode {method}')
    return []


def scorer_version(method: str) -> str | None:
    """Select trained V2 or V3 according to method; controls use no predictor."""
    if method not in MODES:
        raise ValueError(f'Unknown method {method}')
    if method.startswith('predictor_v3_'):
        return 'v3'
    if method.startswith('predictor_large_'):
        return 'v2'
    return None


def render(method: str, gray: np.ndarray, tiles: list, selected: list[int], sr, device):
    if method == 'linear_640':
        return cv2.resize(gray, HR, interpolation=cv2.INTER_LINEAR), 0
    if method == 'full_bicubic_640':
        return cv2.resize(gray, HR, interpolation=cv2.INTER_CUBIC), 0
    if method == 'full_large_fsrcnn_640':
        return render_fsrcnn(gray, tiles, [], sr, device, 'full')
    if method.endswith('_h10'):
        halo = 10
    elif method.endswith('_h16'):
        halo = 16
    else:
        raise ValueError(f'Unknown mode: {method}')
    if len(selected) != 4:
        raise ValueError('Selective modes must enhance exactly four tiles')
    return render_batched_fsrcnn(gray, tiles, selected, sr, device, feather=8, halo=halo)


def synchronize(device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def timed_trial(method, gray, name, tiles, scorers, sr, detector, cfg, device,
                expected: dict[str, tuple[int, ...]]):
    if method not in MODES:
        raise ValueError('Unknown evaluation method')
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter_ns()
    version = scorer_version(method)
    scores = scorers[version](gray) if version else None
    chosen = assigned_tiles(method, name, scores, int(cfg['seed']))
    if tuple(chosen) != expected[method]:
        raise RuntimeError(f'Selection changed during repeats for {method}, image={name}')
    synchronize(device)
    t1 = time.perf_counter_ns()
    output, processed = render(method, gray, tiles, chosen, sr, device)
    synchronize(device)
    t2 = time.perf_counter_ns()
    preds = prediction_records(detector, output, cfg)
    synchronize(device)
    t3 = time.perf_counter_ns()
    return {
        'selector_ms': (t1-t0) / 1e6,
        'sr_ms': (t2-t1) / 1e6,
        'detector_ms': (t3-t2) / 1e6,
        'total_ms': (t3-t0) / 1e6,
        'computed_pixels': processed,
        'peak_torch_mib': torch.cuda.max_memory_allocated(device) / 1024**2,
    }, preds


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('Refusing to save an empty CSV')
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def utility_delta(a: dict, b: dict, fp_penalty: float = .25) -> float:
    """a minus b; positive means more small TP and/or fewer FP."""
    return int(a['small_tp']) - int(b['small_tp']) - fp_penalty * (int(a['fp']) - int(b['fp']))


def cluster_bootstrap_ci(per_image: dict[str, float], by_seq: dict[str, list[str]],
                         seed: int, n_boot: int = 2000) -> list[float]:
    """Sequence-cluster bootstrap; each sampled cluster keeps all its frames.

    Returns a frame-weighted mean delta per image CI (not pseudo-independent
    repeated timings). For a small sequence count this is exploratory only.
    """
    if n_boot < 100 or len(by_seq) < 2:
        raise ValueError('Need >=100 bootstrap samples and >=2 video sequences')
    seqs = sorted(by_seq)
    if set(per_image) != {i for s in seqs for i in by_seq[s]}:
        raise ValueError('Inconsistent image/sequence membership')
    seq_sums = np.asarray([sum(per_image[i] for i in by_seq[s]) for s in seqs], np.float64)
    seq_sizes = np.asarray([len(by_seq[s]) for s in seqs], np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(seqs), size=(n_boot, len(seqs)))
    estimates = seq_sums[picks].sum(axis=1) / seq_sizes[picks].sum(axis=1)
    return [round(float(np.percentile(estimates, p)), 6) for p in (2.5, 97.5)]


def summarize(trials: list[dict], detections: list[dict], seed: int) -> dict:
    if not trials or not detections:
        raise ValueError('No data')
    by_det: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in detections:
        i, m = row['image'], row['method']
        if m not in MODES or m in by_det[i] or row['sequence'] != sequence_id(i):
            raise ValueError('Invalid/duplicate detection row')
        by_det[i][m] = row
    images = sorted(by_det)
    if any(set(method_rows) != set(MODES) for method_rows in by_det.values()):
        raise ValueError('Missing detection methods')
    by_seq: dict[str, list[str]] = defaultdict(list)
    for i in images:
        by_seq[sequence_id(i)].append(i)
    if len(by_seq) < 2:
        raise ValueError('Need at least two video sequences')
    by_trial = {}
    for r in trials:
        key = (r['image'], r['method'], int(r['repeat']))
        if key in by_trial or r['sequence'] != sequence_id(r['image']):
            raise ValueError('Duplicate or inconsistent timing trial')
        by_trial[key] = r
    repeats = sorted({int(r['repeat']) for r in trials})
    expected = {(i,m,k) for i in images for m in MODES for k in repeats}
    if set(by_trial) != expected:
        raise ValueError('Missing/extra timing trials')
    out = {}
    per_image_times = {}
    for m in MODES:
        counts = {key: sum(int(by_det[i][m][key]) for i in images)
                  for key in ('small_gt', 'small_tp', 'all_tp', 'fp')}
        # image-median measured timings limit outlier influence; summary averages images.
        timed = {i: {metric: statistics.median(float(by_trial[i,m,k][metric]) for k in repeats)
                    for metric in ('selector_ms','sr_ms','detector_ms','total_ms')}
                 for i in images}
        per_image_times[m] = timed
        out[m] = {
            **counts,
            'small_recall_conf_025': round(counts['small_tp']/counts['small_gt'], 6) if counts['small_gt'] else None,
            'mean_of_image_median_total_ms': round(statistics.mean(v['total_ms'] for v in timed.values()), 4),
            'mean_selector_ms': round(statistics.mean(v['selector_ms'] for v in timed.values()), 4),
            'mean_sr_ms': round(statistics.mean(v['sr_ms'] for v in timed.values()), 4),
            'mean_detector_ms': round(statistics.mean(v['detector_ms'] for v in timed.values()), 4),
            'mean_processed_lr_pixels': round(statistics.mean(float(by_trial[i,m,repeats[0]]['computed_pixels']) for i in images), 1),
        }
    # All methods must use the same annotated GT.
    if len({out[m]['small_gt'] for m in MODES}) != 1:
        raise ValueError('Small ground truth differs by method')

    comparisons = {}
    comparison_pairs = (
        ('full_large_fsrcnn_640','full_bicubic_640'),
        ('predictor_large_roi4_h16','full_large_fsrcnn_640'),
        ('predictor_v3_large_roi4_h10','predictor_large_roi4_h10'),
        ('predictor_v3_large_roi4_h16','predictor_large_roi4_h16'),
        ('predictor_v3_large_roi4_h10','random_large_roi4_h10'),
        ('predictor_v3_large_roi4_h10','full_large_fsrcnn_640'),
        ('predictor_v3_large_roi4_h10','full_bicubic_640'),
        ('predictor_v3_large_roi4_h10','predictor_v3_large_roi4_h16'),
        ('predictor_large_roi4_h10','full_large_fsrcnn_640'),
        ('predictor_large_roi4_h10','full_bicubic_640'),
        ('predictor_large_roi4_h10','random_large_roi4_h10'),
        ('predictor_large_roi4_h16','random_large_roi4_h16'),
        ('predictor_large_roi4_h10','predictor_large_roi4_h16'),
        ('random_large_roi4_h10','full_large_fsrcnn_640'),
    )
    for index, (a, b) in enumerate(comparison_pairs):
        u = {i: utility_delta(by_det[i][a], by_det[i][b]) for i in images}
        t = {i: per_image_times[a][i]['total_ms'] - per_image_times[b][i]['total_ms'] for i in images}
        comparisons[f'{a}__vs__{b}'] = {
            'delta_small_tp': out[a]['small_tp']-out[b]['small_tp'],
            'delta_fp': out[a]['fp']-out[b]['fp'],
            'delta_utility_total': round(sum(u.values()), 4),
            'delta_utility_per_image': round(statistics.mean(u.values()), 6),
            'utility_per_image_sequence_bootstrap_95ci': cluster_bootstrap_ci(u, by_seq, seed+index*17),
            'delta_total_ms_per_image': round(statistics.mean(t.values()), 4),
            'latency_difference_sequence_bootstrap_95ci_ms': cluster_bootstrap_ci(t, by_seq, seed+index*17+10),
            'sequence_utility_wins_ties_losses': {
                'wins': sum(sum(u[i] for i in by_seq[s]) > 0 for s in by_seq),
                'ties': sum(sum(u[i] for i in by_seq[s]) == 0 for s in by_seq),
                'losses': sum(sum(u[i] for i in by_seq[s]) < 0 for s in by_seq),
            },
        }
    return {
        'images': len(images), 'unique_sequences': len(by_seq), 'repeats_per_image': len(repeats),
        'methods': out, 'comparisons': comparisons,
        'utility_definition': 'delta_smallTP - 0.25 * delta_FP; diagnostic conf=0.25, IoU=0.50',
        'caveats': [
            'Development FLIR VAL already inspected in earlier stages; not an independent final test.',
            'No standard mAP computed here; small TP and FP are class-aware diagnostics.',
            'V2 selected Bicubic-label regions; V3 trained with Large-FSRCNN halo-10 ROI intervention labels.',
            'The 200-image development set has influenced prior model decisions; exploratory analyses only.',
            'Latency excludes disk IO and checkpoint load, synchronizes GPU around stages.',
            'Sequence-group bootstrap on only 17 video IDs; intervals are exploratory.',
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/large_v2_v3_step10d.yaml'))
    p.add_argument('--limit', type=int, default=8, help='0 for all 200 frozen images')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.limit < 0 or not 3 <= a.repeats <= 10 or a.out.exists():
        p.error('Require limit>=0, repeats=3..10 and unused --out directory')
    if not a.config.is_file():
        p.error(f'Missing experiment config: {a.config}')
    exp = yaml.safe_load(a.config.read_text(encoding='utf-8'))
    if not isinstance(exp, dict):
        p.error('Invalid experiment YAML')
    original_path = Path(exp['base_development_config'])
    original = yaml.safe_load(original_path.read_text(encoding='utf-8'))
    frozen_path = Path(exp['frozen_manifest'])
    manifest = json.loads(frozen_path.read_text(encoding='utf-8'))
    validate_protocol(exp, original, manifest)
    val_dir = Path(original['eval_lr'])
    train_dir = Path(original['train_lr'])
    labels = val_dir.parent.parent / 'labels' / 'val'
    training_seq = training_sequences(train_dir, Path(original['utility_csv']), Path(original['split_manifest']))
    names = check_manifest(manifest, val_dir, labels, training_seq,
                           Path(original['utility_csv']), Path(original['split_manifest']))
    # SHA256 checks are NEVER bypassed for frozen Step 7A baseline artifacts.
    for key in ('predictor_checkpoint', 'detector_checkpoint', 'sr_checkpoint'):
        path = Path(original[key])
        if manifest['checkpoint_sha256'][key] != sha256(path):
            raise ValueError(f'Frozen checkpoint hash mismatch: {key}')
    new_ckpts = {
        'large_sr_checkpoint': Path(exp['large_sr_checkpoint']),
        'predictor_v2_checkpoint': Path(exp['predictor_v2_checkpoint']),
        'predictor_v2_training_csv': Path(exp['predictor_v2_training_csv']),
        'predictor_v3_checkpoint': Path(exp['predictor_v3_checkpoint']),
        'predictor_v3_training_csv': Path(exp['predictor_v3_training_csv']),
    }
    for key,path in new_ckpts.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {key}: {path}')
    if a.limit:
        names = names[:a.limit]
    if len(names) < 2 or len({sequence_id(i) for i in names}) < 2:
        p.error('Select at least two frames from two distinct sequences')
    if not torch.cuda.is_available():
        p.error('CUDA is required for the RTX 5070 end-to-end timing protocol')
    cv2.setNumThreads(1)
    device = torch.device('cuda:0')
    predictor_v2, input_size_v2 = predictor_model(
        new_ckpts['predictor_v2_checkpoint'], new_ckpts['predictor_v2_training_csv'], device)
    predictor_v3, input_size_v3 = predictor_model(
        new_ckpts['predictor_v3_checkpoint'], new_ckpts['predictor_v3_training_csv'], device)
    if input_size_v2 != input_size_v3:
        raise ValueError('V2 and V3 predictor input shapes must match')
    sr = load_sr_checkpoint(new_ckpts['large_sr_checkpoint'], device)
    from ultralytics import YOLO
    detector = YOLO(str(original['detector_checkpoint']))
    tiles = roi_grid(640, 512, 4, 4)
    scorers = {
        'v2': FastScorer(predictor_v2, tiles, input_size_v2, device,
                         float(original['false_positive_penalty'])),
        'v3': FastScorer(predictor_v3, tiles, input_size_v3, device,
                         float(original['false_positive_penalty'])),
    }
    dummy = np.zeros(LR_SHAPE, dtype=np.uint8)
    for _ in range(2):
        for scorer in scorers.values():
            scorer(dummy)
        for mode in MODES:
            selected = [0,5,10,15] if '_roi4_' in mode else []
            frame,_ = render(mode, dummy, tiles, selected, sr, device)
            prediction_records(detector, frame, original)
    synchronize(device)
    trials, dets = [], []
    for idx, name in enumerate(names,1):
        gray = cv2.imread(str(val_dir/name), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != LR_SHAPE:
            raise ValueError(f'Invalid 320x256 LR grayscale frame: {name}')
        scores_fast = {version: scorer(gray) for version, scorer in scorers.items()}
        # Smoke: audit top-4 fast selector against original once per image (outside timing).
        if a.limit and a.limit <= 8:
            for version, (model, size) in {
                'v2': (predictor_v2, input_size_v2),
                'v3': (predictor_v3, input_size_v3),
            }.items():
                original_scores = predict_tile_scores(model, gray, tiles, size,
                                     device, float(original['false_positive_penalty']))
                if not np.allclose(original_scores, scores_fast[version], atol=1e-6, rtol=0):
                    raise RuntimeError(f'Fast {version} scorer changed ROI ranking scores')
        expected = {
            mode: tuple(assigned_tiles(mode, name,
                         scores_fast[scorer_version(mode)] if scorer_version(mode) else None,
                         int(original['seed'])))
            for mode in MODES
        }
        first_pass = {}
        for rep in range(a.repeats):
            order = list(MODES)
            random.Random(int(original['seed'])+idx*1073+rep).shuffle(order)
            for mode in order:
                row, preds = timed_trial(mode, gray, name, tiles, scorers, sr,
                                         detector, original, device, expected)
                trials.append({'image':name, 'sequence':sequence_id(name),
                               'repeat':rep, 'method':mode,
                               **{k:round(float(v),6) for k,v in row.items()}})
                if rep == 0:
                    first_pass[mode] = preds
        # Ground truth is not used for selection or SR, only for offline diagnostics.
        gts = ground_truth(labels / (Path(name).stem+'.txt'))
        for mode in MODES:
            dets.append({'image':name, 'sequence':sequence_id(name), 'method':mode,
                         **detection_counts(gts, first_pass[mode], .25)})
        if idx==1 or idx%10==0 or idx==len(names):
            print(f'Evaluated {idx}/{len(names)} frozen FLIR development frames', flush=True)
    result = summarize(trials, dets, int(original['seed']))
    report = {
        'task': 'Step 10D same-run V2 vs V3 end-to-end Large FSRCNN ROI comparison',
        'method_labels': {
            'predictor_large_roi4_h10': 'V2, Large SR halo 10',
            'predictor_large_roi4_h16': 'V2, Large SR halo 16',
            'predictor_v3_large_roi4_h10': 'V3, Large SR halo 10',
            'predictor_v3_large_roi4_h16': 'V3, Large SR halo 16',
        },
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'experiment_config':str(a.config), 'experiment_config_sha256':sha256(a.config),
        'base_config':str(original_path), 'base_config_sha256':sha256(original_path),
        'split_manifest':str(frozen_path), 'split_manifest_sha256':sha256(frozen_path),
        'frozen_artifact_checks_verified': True,
        'frozen_checkpoints':{key:{'path':original[key],'sha256':sha256(Path(original[key]))}
                              for key in ('predictor_checkpoint','detector_checkpoint','sr_checkpoint')},
        'evaluated_artifacts':{key:{'path':str(path),'sha256':sha256(path)}
                               for key,path in new_ckpts.items()},
        'protocol': manifest['protocol'],
        'results': result,
    }
    a.out.mkdir(parents=True)
    write_csv(a.out/'timing_per_trial.csv', trials)
    write_csv(a.out/'detection_per_image.csv', dets)
    (a.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
    print('\nSTEP 10D — PAIRED V2/V3 DEVELOPMENT RESULTS ONLY (not independent final testing):')
    for m in MODES:
        v = result['methods'][m]
        print(f'  {m:29s} recall={v["small_recall_conf_025"]:.3f} '
              f'FP={v["fp"]:5d} time={v["mean_of_image_median_total_ms"]:.3f} ms')
    print(f'SAVED — {a.out/"summary.json"}',flush=True)


if __name__ == '__main__':
    main()
