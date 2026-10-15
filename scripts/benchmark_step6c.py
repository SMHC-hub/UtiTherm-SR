"""Step 6C1: repeated, synchronized GPU benchmark of Step 6B policies.

Uses the existing Step 6B exact ROI implementation and Step 6A detector +
utility predictor. NO model training, held-out TEST use, or parameter tuning.

Example:
  python scripts/benchmark_step6c.py --limit 4 --repeats 3 \
    --out reports/utility/efficiency_smoke_v1
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
from benchmark_roi_bicubic import MODES, make_frame
from evaluate_selective_sr import (LR_SHAPE, REQUIRED, choose_tiles,
                                   get_eval_image_names, predict_tile_scores,
                                   predictor_model)
from probe_roi_utility import roi_grid, prediction_records, detection_counts
from evaluate_bicubic_baseline import ground_truth


def percentile(numbers: list[float], pct: float) -> float:
    if not numbers or not 0 <= pct <= 100:
        raise ValueError('Require nonempty sample and percentile in [0,100]')
    return float(np.percentile(np.asarray(numbers, dtype=float), pct))


def per_image_medians(rows: list[dict], field: str) -> dict[str, dict[str, float]]:
    """Collapse repeated measurements before aggregating independent images."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row['method'])][str(row['image'])].append(float(row[field]))
    return {method: {image: statistics.median(vals) for image, vals in ims.items()}
            for method, ims in grouped.items()}


def summarize_benchmark(rows: list[dict], seed: int = 42, bootstrap: int = 2000) -> dict:
    if not rows or bootstrap < 100:
        raise ValueError('Need benchmark rows and >=100 bootstrap resamples')
    metrics = ('selector_ms','prep_ms','detector_ms','total_ms')
    medians = {key: per_image_medians(rows, key) for key in metrics}
    images = sorted({row['image'] for row in rows})
    if any(set(medians[key]) != set(MODES) for key in metrics):
        raise ValueError('Not every benchmark method was measured')
    if any(set(medians[key][method]) != set(images) for key in metrics for method in MODES):
        raise ValueError('All methods must evaluate the same images')
    if len({row['repeat'] for row in rows}) * len(images) * len(MODES) != len(rows):
        raise ValueError('Incomplete or duplicate per-image repeat measurements')
    result = {}
    for method in MODES:
        stage_stats = {}
        for key in metrics:
            v = [medians[key][method][image] for image in images]
            stage_stats[key] = {
                'mean_of_image_medians': round(statistics.mean(v), 4),
                'median_of_image_medians': round(statistics.median(v), 4),
                'p95_of_image_medians': round(percentile(v, 95), 4),
            }
        result[method] = stage_stats
    comparisons = {}
    baseline = medians['total_ms']
    for other in ('fullframe_selective4','full_bicubic_640','linear_640'):
        diffs = np.asarray([baseline[other][img] - baseline['roi_only_selective4'][img]
                            for img in images], dtype=np.float64)
        rng = np.random.default_rng(seed)
        means = [float(np.mean(rng.choice(diffs, len(diffs), replace=True)))
                 for _ in range(bootstrap)]
        comparisons[f'{other}_minus_roi_only_selective4_ms'] = {
            'mean_paired_difference_ms': round(float(diffs.mean()), 4),
            'bootstrap_95ci_mean_difference_ms': [round(percentile(means, 2.5), 4),
                                                   round(percentile(means, 97.5), 4)],
            'note': 'Positive difference means ROI-only took less time; exploratory image bootstrap.'
        }
    return {'images': len(images), 'methods': result, 'paired_comparisons': comparisons}


def validate_config(cfg: dict):
    if not isinstance(cfg, dict) or REQUIRED - set(cfg):
        raise ValueError('Missing Step 6A configuration fields')
    if (int(cfg['budget_tiles']) != 4 or int(cfg['feather_hr_px']) != 8 or
        int(cfg['imgsz']) != 640 or str(cfg['device']) != '0' or
        abs(float(cfg['confidence']) - .25) > 1e-9 or
        abs(float(cfg['nms_iou']) - .7) > 1e-9):
        raise ValueError('Step 6C requires the unchanged Step 6B detector / ROI settings')
    lr_root = Path(cfg['train_lr'])
    if (lr_root.name != 'train' or lr_root.parent.name != 'images' or
            'yolo_flir_lr_x2' not in lr_root.parts):
        raise ValueError('Benchmark must remain within detector TRAIN domain')
    for key in ('predictor_checkpoint','detector_checkpoint','utility_csv','split_manifest'):
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(cfg[key])
    if not lr_root.is_dir():
        raise FileNotFoundError(lr_root)
    return lr_root


def run_trial(method: str, gray: np.ndarray, image_name: str, model, input_size,
              detector, cfg: dict, tiles: list, device: torch.device,
              expected_chosen: tuple[int, ...]):
    """Measure full sequential selector+render+YOLO, synchronized at boundaries."""
    if method not in MODES:
        raise ValueError(f'Unknown method {method}')
    torch.cuda.synchronize(device)
    t0 = time.perf_counter_ns()
    if method in ('fullframe_selective4', 'roi_only_selective4'):
        scores = predict_tile_scores(model, gray, tiles, input_size, device,
                                     float(cfg['false_positive_penalty']))
        chosen = tuple(choose_tiles('predictor4', image_name, 4, 16,
                                    int(cfg['seed']), scores))
        if chosen != expected_chosen:
            raise RuntimeError('Selector changed chosen regions across repeated trials')
        torch.cuda.synchronize(device)
        t1 = time.perf_counter_ns()
    else:
        chosen = ()
        t1 = t0  # No selector: do not misattribute synchronization overhead to selection.
    frame, pixels = make_frame(gray, tiles, list(chosen), method, 8, 2)
    torch.cuda.synchronize(device)
    t2 = time.perf_counter_ns()
    preds = prediction_records(detector, frame, cfg)
    torch.cuda.synchronize(device)
    t3 = time.perf_counter_ns()
    return {
        'selector_ms': (t1-t0)/1e6,
        'prep_ms': (t2-t1)/1e6,
        'detector_ms': (t3-t2)/1e6,
        'total_ms': (t3-t0)/1e6,
        'cubic_pixels': pixels,
        'selected_tiles': ' '.join(map(str, chosen)),
    }, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=Path('configs/selective_sr_eval.yaml'))
    ap.add_argument('--limit', type=int, default=4, help='Predictor validation images, 0=all 20')
    ap.add_argument('--repeats', type=int, default=5, help='Timed complete pipeline repeats per image/method')
    ap.add_argument('--warmup', type=int, default=3, help='GPU warmup inference calls, not timed')
    ap.add_argument('--out', type=Path, default=Path('reports/utility/efficiency_smoke_v1'))
    args = ap.parse_args()
    if args.limit < 0 or not 3 <= args.repeats <= 20 or not 2 <= args.warmup <= 20:
        ap.error('Require limit >=0, repeats 3-20, warmup 2-20')
    if args.out.exists():
        ap.error(f'Output directory already exists: {args.out}')
    if not args.config.is_file():
        ap.error(f'Missing config: {args.config}')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    try:
        lr_root = validate_config(cfg)
        names = get_eval_image_names(Path(cfg['utility_csv']), Path(cfg['split_manifest']), args.limit)
    except (ValueError, FileNotFoundError, KeyError) as e:
        ap.error(str(e))
    if not names:
        ap.error('No eligible images')
    if not torch.cuda.is_available():
        ap.error('CUDA GPU is required for matched device benchmarking')
    device = torch.device('cuda:0')
    cv2.setNumThreads(1)
    model, input_size = predictor_model(Path(cfg['predictor_checkpoint']),
                                        Path(cfg['utility_csv']), device)
    from ultralytics import YOLO
    detector = YOLO(str(cfg['detector_checkpoint']))
    tiles = roi_grid(640,512,4,4)
    # Load and warm both networks before any measurement.
    warm_gray = np.zeros(LR_SHAPE, dtype=np.uint8)
    for _ in range(args.warmup):
        scores = predict_tile_scores(model, warm_gray, tiles, input_size, device,
                                     float(cfg['false_positive_penalty']))
        chosen = choose_tiles('predictor4', 'warm.png', 4, 16, int(cfg['seed']), scores)
        for method in MODES:
            frame, _ = make_frame(warm_gray, tiles, chosen, method, 8, 2)
            prediction_records(detector, frame, cfg)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    rows, detection_rows = [], []
    labels = lr_root.parent.parent/'labels'/'train'
    for i, name in enumerate(names, 1):
        gray = cv2.imread(str(lr_root/name), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != LR_SHAPE:
            raise ValueError(f'Expected 320x256 LR image: {name}')
        scores = predict_tile_scores(model, gray, tiles, input_size, device,
                                     float(cfg['false_positive_penalty']))
        selected = tuple(choose_tiles('predictor4', name, 4, 16, int(cfg['seed']), scores))
        # Guarantee *identical* selective outputs; do not accept changed detection conditions.
        full, _ = make_frame(gray, tiles, list(selected), 'fullframe_selective4', 8, 2)
        crop, _ = make_frame(gray, tiles, list(selected), 'roi_only_selective4', 8, 2)
        if not np.array_equal(full, crop):
            raise RuntimeError(f'ROI output no longer pixel-exact in {name}')
        gts = None
        label_path = labels/(Path(name).stem+'.txt')
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        # Only use ground truth AFTER prediction/selection, for offline metrics.
        for repeat in range(args.repeats):
            order = list(MODES)
            random.Random(int(cfg['seed']) + i*1009 + repeat).shuffle(order)
            for method in order:
                measures, preds = run_trial(method, gray, name, model, input_size,
                                            detector, cfg, tiles, device, selected)
                rows.append({'image': name,
                             'sequence': Path(name).stem.split('-frame-')[0],
                             'repeat': repeat,
                             'method': method, **{k:round(v,5) if isinstance(v, float) else v
                                                   for k,v in measures.items()}})
                if repeat == 0:
                    if gts is None:
                        gts = ground_truth(label_path)
                    cnt = detection_counts(gts, preds, float(cfg['confidence']))
                    detection_rows.append({'image': name, 'method': method, **cnt})
        print(f'Benchmarked {i}/{len(names)} images; {args.repeats} paired randomized repeats; ROI pixels EXACT', flush=True)
    benchmark = summarize_benchmark(rows, seed=int(cfg['seed']))
    by_method = {}
    for method in MODES:
        group = [x for x in detection_rows if x['method'] == method]
        by_method[method] = {k:sum(row[k] for row in group)
                             for k in ('small_tp','small_gt','fp')}
    if by_method['fullframe_selective4'] != by_method['roi_only_selective4']:
        raise RuntimeError('Detection diagnostics inconsistent between pixel-exact ROI and full-frame modes')
    args.out.mkdir(parents=True)
    def save_csv(path: Path, records: list[dict]):
        with path.open('w', newline='', encoding='utf-8') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    save_csv(args.out/'timing_per_trial.csv', rows)
    save_csv(args.out/'detection_first_pass.csv', detection_rows)
    summary = {
        'task': 'UtiTherm-SR Step 6C1 controlled repeated synchronized end-to-end benchmark',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'predictor_checkpoint': str(cfg['predictor_checkpoint']),
        'detector_checkpoint': str(cfg['detector_checkpoint']),
        'split': 'Predictor validation sequences within YOLO detector TRAIN domain',
        'repeats_per_image_policy': args.repeats,
        'warmup_calls': args.warmup,
        'device': torch.cuda.get_device_name(device),
        'max_gpu_allocated_mib_during_benchmark': round(torch.cuda.max_memory_allocated(device)/1024**2, 2),
        'timings': benchmark,
        'detection_diagnostic_first_pass': by_method,
        'roi_pixel_exactness': True,
        'notes': [
            'Same 640 detector resolution and fixed checkpoints in all conditions.',
            'Stages use synchronized wall-clock timings including Python, preprocessing, model dispatch, prediction and CPU output transfer.',
            'Data loading and annotation matching excluded from timing, but YOLO/selector preprocessing included.',
            'Methods run in randomized order each repeat; warmup precedes timing; image-repeated samples aggregated first.',
            'Per-image bootstrap CI is exploratory and small (20 images max), not a final performance guarantee.',
            'No separate time saving from learned SR established; cropped Bicubic only.',
            'Checkpoint selected on these predictor-held-out sequences; detector TRAIN domain; no independent test.',
            'These results must not be compared as directly equivalent to unstandardized previous-run latency.',
        ]
    }
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print('\nSTEP 6C1 SPEED RESULTS — mean of per-image medians, milliseconds/image:')
    for mode in MODES:
        stage = benchmark['methods'][mode]
        print(f"  {mode:24s} total={stage['total_ms']['mean_of_image_medians']:.3f} "
              f"selector={stage['selector_ms']['mean_of_image_medians']:.3f} "
              f"prepare={stage['prep_ms']['mean_of_image_medians']:.3f} "
              f"detector={stage['detector_ms']['mean_of_image_medians']:.3f}")
    print('Paired timing differences (positive means ROI-only faster):')
    for comparison, value in benchmark['paired_comparisons'].items():
        print(f"  {comparison}: mean={value['mean_paired_difference_ms']:+.3f}, "
              f"95% exploratory CI={value['bootstrap_95ci_mean_difference_ms']}")
    print(f'DONE — {args.out / "summary.json"}; {args.out / "timing_per_trial.csv"}',flush=True)


if __name__ == '__main__':
    main()
