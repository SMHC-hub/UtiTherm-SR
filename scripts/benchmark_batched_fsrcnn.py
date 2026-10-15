"""Step 6C3: paired timing of serial vs ONE-batch ROI FSRCNN.

No dataset changes or retraining. Uses the Step 5B predictor-held-out
subset from detector-TRAIN-domain data, not an unbiased independent test.

    python scripts/benchmark_batched_fsrcnn.py --config configs/selective_sr_eval.yaml \
      --limit 4 --repeats 3 --out reports/utility/fsrcnn_batched_smoke_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

from utitherm.sr.roi_fsrcnn import selected_region_difference
from utitherm.sr.roi_fsrcnn_batched import render_batched_fsrcnn
from benchmark_selective_fsrcnn import (
    validate_settings, load_sr_checkpoint, render as old_render,
)
from evaluate_selective_sr import (
    choose_tiles, get_eval_image_names, predict_tile_scores, predictor_model,
)
from probe_roi_utility import roi_grid, prediction_records, detection_counts
from evaluate_bicubic_baseline import ground_truth

MODES = (
    'linear_640', 'full_bicubic_640', 'full_fsrcnn_640',
    'selective_fullframe_fsrcnn4', 'selective_roi_fsrcnn4',
    'selective_batched_roi_fsrcnn4',
)
LR_SHAPE = (256, 320)


def render(mode: str, gray: np.ndarray, tiles: list, chosen: list[int],
           sr, device: torch.device, halo: int) -> tuple[np.ndarray, int]:
    if mode == 'selective_batched_roi_fsrcnn4':
        return render_batched_fsrcnn(gray, tiles, chosen, sr, device,
                                     feather=8, halo=halo)
    if mode not in MODES:
        raise ValueError(f'Unknown policy: {mode}')
    return old_render(mode, gray, tiles, chosen, sr, device, halo)


def time_trial(mode: str, gray: np.ndarray, name: str, tiles: list,
               predictor, input_size, sr, detector, cfg: dict,
               device: torch.device, expected: tuple[int, ...],
               halo: int) -> tuple[dict, list[dict]]:
    """Time selector, SR, detector as one synchronized measurement."""
    if mode not in MODES:
        raise ValueError(f'Unknown policy: {mode}')
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter_ns()
    chosen: list[int] = []
    if mode.startswith('selective_'):
        scores = predict_tile_scores(predictor, gray, tiles, input_size, device,
                                     float(cfg['false_positive_penalty']))
        chosen = choose_tiles('predictor4', name, 4, len(tiles),
                              int(cfg['seed']), scores)
        if tuple(chosen) != expected:
            raise RuntimeError('Predictor rankings changed during trials')
        torch.cuda.synchronize(device)
    t1 = time.perf_counter_ns()
    frame, pixels = render(mode, gray, tiles, chosen, sr, device, halo)
    torch.cuda.synchronize(device)
    t2 = time.perf_counter_ns()
    preds = prediction_records(detector, frame, cfg)
    torch.cuda.synchronize(device)
    t3 = time.perf_counter_ns()
    return {
        'selector_ms': (t1-t0)/1e6,
        'sr_or_resize_ms': (t2-t1)/1e6,
        'detector_ms': (t3-t2)/1e6,
        'total_ms': (t3-t0)/1e6,
        'model_input_lr_pixels': pixels,
        'pytorch_peak_allocated_mib': torch.cuda.max_memory_allocated(device)/(1024**2),
    }, preds


def summarize_trials(rows: list[dict], seed: int = 42,
                     bootstraps: int = 1000) -> dict:
    """Use image as resampling unit; repeated timings aren't new samples."""
    if not rows or bootstraps < 100:
        raise ValueError('Need benchmark trials and >=100 bootstrap resamples')
    names = sorted({r['image'] for r in rows})
    reps = sorted({int(r['repeat']) for r in rows})
    expected = {(n, m, r) for n in names for m in MODES for r in reps}
    by_key = {}
    for row in rows:
        key = (row['image'], row['method'], int(row['repeat']))
        if key in by_key or key not in expected:
            raise ValueError(f'Unexpected or duplicate benchmark trial: {key}')
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError('Incomplete benchmark trials')
    fields = ('selector_ms','sr_or_resize_ms','detector_ms','total_ms',
              'model_input_lr_pixels','pytorch_peak_allocated_mib')
    by_mode = {}; timings = {}
    for mode in MODES:
        per_field = {}
        for field in fields:
            samples = [statistics.median(float(by_key[(n, mode, r)][field]) for r in reps)
                       for n in names]
            per_field[field] = {'mean_of_per_image_medians': round(statistics.mean(samples), 4),
                                'median_of_per_image_medians': round(statistics.median(samples), 4)}
            if field == 'total_ms':
                timings[mode] = np.array(samples)
        by_mode[mode] = per_field
    rng = np.random.default_rng(seed)
    comparisons = {}
    for other in ('selective_roi_fsrcnn4', 'selective_fullframe_fsrcnn4',
                  'full_fsrcnn_640', 'full_bicubic_640'):
        diffs = timings[other] - timings['selective_batched_roi_fsrcnn4']
        boot = np.array([rng.choice(diffs, len(diffs), replace=True).mean()
                         for _ in range(bootstraps)])
        comparisons[f'{other}_minus_batched_ms'] = {
            'mean_paired_difference_ms': round(float(diffs.mean()), 4),
            'exploratory_image_bootstrap_95ci': [
                round(float(np.percentile(boot, 2.5)), 4),
                round(float(np.percentile(boot, 97.5)), 4)],
            'positive_means_batched_is_faster': True,
        }
    return {'images': len(names), 'repeats': len(reps),
            'methods': by_mode, 'paired_comparisons': comparisons}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('Empty CSV records')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/selective_sr_eval.yaml'))
    parser.add_argument('--sr-checkpoint', type=Path, default=Path('models/sr/fsrcnn_x2_v1.pt'))
    parser.add_argument('--limit', type=int, default=4, help='0=all 20 eligible images')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--halo', type=int, default=16)
    parser.add_argument('--max-diff', type=int, default=1)
    parser.add_argument('--out', type=Path, default=Path('reports/utility/fsrcnn_batched_smoke_v1'))
    args = parser.parse_args()
    if args.limit < 0 or not 3 <= args.repeats <= 20 or not 2 <= args.warmup <= 20:
        parser.error('Require limit>=0, repeats=3..20 and warmup=2..20')
    if args.halo < 10 or args.halo > 64 or not 0 <= args.max_diff <= 2:
        parser.error('Require halo=10..64 and max-diff=0..2')
    if args.out.exists():
        parser.error(f'Experiment directory exists: {args.out}; choose a new --out')
    if not args.config.is_file():
        parser.error(f'Config missing: {args.config}')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    validate_settings(cfg)
    source = Path(cfg['train_lr'])
    for p in (source, Path(cfg['utility_csv']), Path(cfg['split_manifest']),
              Path(cfg['predictor_checkpoint']), Path(cfg['detector_checkpoint']),
              args.sr_checkpoint):
        if not p.exists():
            parser.error(f'Missing required project asset: {p}')
    names = get_eval_image_names(Path(cfg['utility_csv']),
                                 Path(cfg['split_manifest']), args.limit)
    if not names:
        parser.error('No eligible held-out images')
    labels = source.parent.parent/'labels'/'train'
    if not labels.is_dir():
        parser.error(f'Missing label directory: {labels}')
    if not torch.cuda.is_available():
        parser.error('CUDA GPU required for the speed benchmark')
    device = torch.device('cuda:0')
    cv2.setNumThreads(1)
    predictor, input_size = predictor_model(Path(cfg['predictor_checkpoint']),
                                            Path(cfg['utility_csv']), device)
    sr = load_sr_checkpoint(args.sr_checkpoint, device)
    from ultralytics import YOLO
    detector = YOLO(str(cfg['detector_checkpoint']))
    tiles = roi_grid(640, 512, 4, 4)
    warm = np.zeros(LR_SHAPE, np.uint8)
    for _ in range(args.warmup):
        predict_tile_scores(predictor, warm, tiles, input_size,
                            device, float(cfg['false_positive_penalty']))
        for mode in MODES:
            frame, _ = render(mode, warm, tiles, [0,5,10,15], sr, device, args.halo)
            prediction_records(detector, frame, cfg)
    torch.cuda.synchronize(device)

    trials = []
    detections = []
    audits = []
    for k, name in enumerate(names, start=1):
        gray = cv2.imread(str(source/name), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != LR_SHAPE:
            raise ValueError(f'Expected 320x256 grayscale image: {name}')
        label_path = labels/(Path(name).stem+'.txt')
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        scores = predict_tile_scores(predictor, gray, tiles, input_size,
                                     device, float(cfg['false_positive_penalty']))
        chosen = tuple(choose_tiles('predictor4', name, 4, len(tiles),
                                    int(cfg['seed']), scores))
        ref, _ = render('selective_fullframe_fsrcnn4', gray, tiles,
                        list(chosen), sr, device, args.halo)
        serial, n_serial = render('selective_roi_fsrcnn4', gray, tiles,
                                  list(chosen), sr, device, args.halo)
        batched, n_batched = render('selective_batched_roi_fsrcnn4', gray, tiles,
                                    list(chosen), sr, device, args.halo)
        diff_ref = selected_region_difference(ref, batched, tiles, list(chosen))
        diff_serial = selected_region_difference(serial, batched, tiles, list(chosen))
        if max(diff_ref['max_abs_difference'], diff_serial['max_abs_difference']) > args.max_diff:
            raise RuntimeError(f'FSRCNN ROI equivalence failed on {name}: '
                               f'full={diff_ref}, serial={diff_serial}')
        audits.append({
            'image': name, 'selected_tiles': ' '.join(map(str, chosen)),
            'reference_max_pixel_diff': diff_ref['max_abs_difference'],
            'serial_max_pixel_diff': diff_serial['max_abs_difference'],
            'changed_output_pixels_vs_reference': diff_ref['changed_output_pixels'],
            'serial_lr_model_pixels': n_serial,
            'batched_lr_model_pixels': n_batched,
            'batched_lr_input_fraction': round(n_batched / gray.size, 6),
        })
        gt = ground_truth(label_path)
        for rep in range(args.repeats):
            order = list(MODES)
            random.Random(int(cfg['seed'])+k*1009+rep).shuffle(order)
            for mode in order:
                measured, pred = time_trial(mode, gray, name, tiles, predictor,
                                            input_size, sr, detector, cfg, device,
                                            chosen, args.halo)
                trials.append({'image': name, 'repeat': rep, 'method': mode,
                               **{key: round(value, 6) if isinstance(value, float) else value
                                  for key, value in measured.items()}})
                if rep == 0:
                    detections.append({'image': name, 'method': mode,
                                       **detection_counts(gt, pred, float(cfg['confidence']))})
        print(f'Benchmarked {k}/{len(names)} | batched-vs-full max diff='
              f'{diff_ref["max_abs_difference"]} | batched-vs-serial max diff='
              f'{diff_serial["max_abs_difference"]} | batched input '
              f'{n_batched/gray.size:.1%}', flush=True)

    timing = summarize_trials(trials, seed=int(cfg['seed']))
    aggregate = {}
    for mode in MODES:
        data = [r for r in detections if r['method'] == mode]
        counts = {field: sum(int(r[field]) for r in data)
                  for field in ('small_gt','small_tp','all_tp','fp')}
        aggregate[mode] = {**counts, 'small_recall_conf_025': (
            round(counts['small_tp']/counts['small_gt'],6) if counts['small_gt'] else None)}
    args.out.mkdir(parents=True)
    write_csv(args.out/'timing_per_trial.csv', trials)
    write_csv(args.out/'detection_per_image.csv', detections)
    write_csv(args.out/'roi_equivalence.csv', audits)
    checkpoints = {key: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                   for key, p in {'sr': args.sr_checkpoint,
                                  'detector': Path(cfg['detector_checkpoint']),
                                  'predictor': Path(cfg['predictor_checkpoint'])}.items()}
    summary = {
        'task': 'UtiTherm-SR Step 6C3 batched ROI neural super-resolution',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'images': len(names), 'repeats': args.repeats, 'halo_lr_px': args.halo,
        'allowed_roi_max_abs_pixel_difference': args.max_diff,
        'max_observed_full_vs_batch_difference': max(x['reference_max_pixel_diff'] for x in audits),
        'max_observed_serial_vs_batch_difference': max(x['serial_max_pixel_diff'] for x in audits),
        'mean_batched_lr_model_input_fraction': round(statistics.mean(
            x['batched_lr_input_fraction'] for x in audits), 6),
        'checkpoints': checkpoints,
        'timings_ms': timing,
        'detection': aggregate,
        'timing_excludes': 'disk I/O, annotation loading, dataset write, model loading',
        'source': 'predictor-held-out detector-TRAIN-domain images; NOT independent test',
        'warnings': [
            'Predictor was trained on Bicubic utility, not FSRCNN intervention utility.',
            'The predictor checkpoint was selected on this internal validation domain.',
            'The already-trained detector has seen images from this TRAIN domain.',
            'The batched crop method performs one FSRCNN forward pass for the 4 selected windows.',
            'Shifted uniform crops may use more LR model input pixels than clipped sequential crops.',
            'Pixel tolerance is in grayscale intensity levels; changes in YOLO predictions must be reported.',
            'No inference-time ground truth used in region selection; GT used only for metrics.',
            'Reported GPU memory is peak PyTorch allocation, not actual power or GPU utilization.',
            'Repetitions are timings of same images; exploratory bootstrap resamples unique images.',
            'No official COCO AP in this benchmark; diagnostic TP/FP IoU=.5, conf=.25 only.',
            'Do not make efficiency or generalization claims before independent holdout verification.',
        ],
    }
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print('\nSTEP 6C3 SPEED RESULTS — mean of per-image timing medians (ms/image):')
    for mode in MODES:
        m = timing['methods'][mode]
        det = aggregate[mode]
        print(f'  {mode:31s} total={m["total_ms"]["mean_of_per_image_medians"]:7.3f} '
              f'select={m["selector_ms"]["mean_of_per_image_medians"]:6.3f} '
              f'SR={m["sr_or_resize_ms"]["mean_of_per_image_medians"]:6.3f} '
              f'smallTP={det["small_tp"]} FP={det["fp"]}')
    for key, obj in timing['paired_comparisons'].items():
        print(f'  {key}: {obj["mean_paired_difference_ms"]:+.3f} ms '
              f'CI={obj["exploratory_image_bootstrap_95ci"]}')
    print(f'DONE — {args.out/"summary.json"}; {args.out/"timing_per_trial.csv"}',flush=True)


if __name__ == '__main__':
    main()
