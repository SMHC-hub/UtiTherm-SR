"""Step 6A: actual multi-ROI detection test on predictor-held-out TRAIN-domain frames.

Compared policies use identical 640x512 detector input, fixed YOLO checkpoint,
IoU=0.50 and conf=0.25 for diagnostic TP/FP counts. ROI selector never sees GT.

Run:
  python scripts/evaluate_selective_sr.py --config configs/selective_sr_eval.yaml \
    --limit 4 --out reports/utility/selective_smoke_v1

Run full 20-image predictor-held-out pilot with optional standardized mAP:
  python scripts/evaluate_selective_sr.py --config configs/selective_sr_eval.yaml \
    --limit 0 --with-map --out reports/utility/selective_holdout_v1

NOTE: cv2.INTER_CUBIC is currently calculated on the WHOLE frame first and
then blended within selected tiles, exactly like Step 5A. This does not save
bicubic interpolation compute. It measures selective region *effects* only.
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
from utitherm.utility.predictor import HEADS, UtilityPredictor, encode_tile, ranking_score
from evaluate_bicubic_baseline import ground_truth
from probe_roi_utility import (roi_grid, enhanced_tile, prediction_records,
                               detection_counts, sequence_id)

REQUIRED = {
    'train_lr', 'utility_csv', 'split_manifest', 'predictor_checkpoint',
    'detector_checkpoint', 'budget_tiles', 'seed', 'imgsz', 'confidence',
    'nms_iou', 'false_positive_penalty', 'feather_hr_px', 'val_batch', 'device',
}
POLICIES = ('linear_640', 'fixed_first4', 'random4', 'predictor4', 'full_bicubic_640')
HR_SIZE = (640, 512)
LR_SHAPE = (256, 320)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_eval_image_names(csv_path: Path, split_path: Path, limit: int = 0) -> list[str]:
    """Use only the Step 5B predictor-validation sequences; never detector VAL."""
    if limit < 0:
        raise ValueError('limit must be >= 0')
    if not csv_path.is_file() or not split_path.is_file():
        raise FileNotFoundError('Missing utility CSV or predictor split manifest')
    manifest = json.loads(split_path.read_text(encoding='utf-8'))
    if manifest.get('intervention_csv_sha256') != sha256(csv_path):
        raise ValueError('Split manifest does not match ROI measurement CSV (SHA256 mismatch)')
    val_seqs = set(manifest['val_sequences'])
    train_seqs = set(manifest['train_sequences'])
    if not val_seqs or val_seqs & train_seqs:
        raise ValueError('Predictor training/validation sequences overlap or are empty')
    images_by_seq: dict[str, set[str]] = defaultdict(set)
    with csv_path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'image', 'sequence', 'tile_index'}
        if not reader.fieldnames or required - set(reader.fieldnames):
            raise ValueError('Utility CSV missing image / sequence / tile_index')
        seen: set[tuple[str, int]] = set()
        for r in reader:
            seq, image = r['sequence'], r['image']
            if not seq or seq != sequence_id(Path(image)) or Path(image).name != image or not image.endswith('.png'):
                raise ValueError(f'Unsafe or inconsistent image record: {image}')
            if seq in val_seqs:
                idx = int(r['tile_index'])
                if idx not in range(16) or (image, idx) in seen:
                    raise ValueError(f'Duplicate or invalid tile record: {image} {idx}')
                seen.add((image, idx))
                images_by_seq[seq].add(image)
    if set(images_by_seq) != val_seqs:
        raise ValueError('Predictor validation sequences missing from ROI labels')
    # Restrict to *exactly* the labeled images, with complete 16-tile interventions.
    images = sorted({img for group in images_by_seq.values() for img in group})
    if any(sum((img, i) in seen for i in range(16)) != 16 for img in images):
        raise ValueError('At least one image does not have all 16 ROI measurements')
    return images[:limit] if limit else images


def predictor_model(ckpt: Path, csv_path: Path, device: torch.device):
    if not ckpt.is_file():
        raise FileNotFoundError(f'Missing predictor checkpoint: {ckpt}')
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    if tuple(state['heads']) != HEADS or list(state['input_size']) != [80, 64]:
        raise ValueError('Predictor architecture/encoding does not match Step 5B')
    if state.get('source_csv_sha256') != sha256(csv_path):
        raise ValueError('Predictor trained on a different ROI measurement CSV')
    model = UtilityPredictor().to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    return model, tuple(state['input_size'])


def predict_tile_scores(model, gray: np.ndarray, tiles: list[tuple[int,int,int,int]],
                        input_size: tuple[int,int], device: torch.device,
                        fp_penalty: float) -> np.ndarray:
    """Only LR pixels and fixed grid coordinates enter predictor inference."""
    encoded = [encode_tile(gray, rect, input_size=input_size, hr_size=HR_SIZE) for rect in tiles]
    x = torch.stack([a for a, _ in encoded]).to(device)
    coords = torch.stack([b for _, b in encoded]).to(device)
    with torch.inference_mode():
        probabilities = model(x, coords).sigmoid().cpu().numpy()
    scores = np.asarray(ranking_score(probabilities, fp_penalty), dtype=np.float64)
    if scores.shape != (len(tiles),) or not np.isfinite(scores).all():
        raise ValueError('Invalid utility predictor output')
    return scores


def choose_tiles(policy: str, image_name: str, budget: int, count: int,
                 seed: int, scores: np.ndarray | None = None) -> list[int]:
    if not 1 <= budget <= count:
        raise ValueError('Tile budget is outside grid bounds')
    if policy == 'fixed_first4':
        return list(range(budget))
    if policy == 'random4':
        # Stable for each (image,seed), independent of Python hash randomization.
        digest = hashlib.sha256(f'{seed}:{image_name}'.encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], 'little'))
        return sorted(rng.sample(range(count), budget))
    if policy == 'predictor4':
        if scores is None or len(scores) != count or not np.isfinite(scores).all():
            raise ValueError('Need finite score for every tile')
        return sorted(range(count), key=lambda idx: (-float(scores[idx]), idx))[:budget]
    raise ValueError(f'Not a region-selection policy: {policy}')


def apply_tiles(base: np.ndarray, cubic: np.ndarray,
                tiles: list[tuple[int,int,int,int]], selected: list[int],
                feather: int) -> np.ndarray:
    if not selected or len(selected) != len(set(selected)) or any(i < 0 or i >= len(tiles) for i in selected):
        raise ValueError('Invalid selected tile list')
    out = base.copy()
    for i in selected:
        # Reuse the EXACT Step 5A feathered intervention definition.
        # Tile interiors do not overlap, so sequential composition is well-defined.
        out = enhanced_tile(out, cubic, tiles[i], feather)
    return out


def build_policy_image(policy: str, gray: np.ndarray, tiles: list,
                       budget: int, seed: int, image_name: str, feather: int,
                       model, input_size: tuple[int,int], device: torch.device,
                       fp_penalty: float) -> tuple[np.ndarray, list[int], float]:
    """Returns (HR image, chosen tile indices, predictor time). Does not read GT."""
    if gray.shape != LR_SHAPE or gray.dtype != np.uint8:
        raise ValueError('Expect grayscale LR frame 320x256 uint8')
    base = cv2.resize(gray, HR_SIZE, interpolation=cv2.INTER_LINEAR)
    if policy == 'linear_640':
        return base, [], 0.0
    cubic = cv2.resize(gray, HR_SIZE, interpolation=cv2.INTER_CUBIC)
    if policy == 'full_bicubic_640':
        return cubic, list(range(len(tiles))), 0.0
    selector_ms = 0.0
    scores = None
    if policy == 'predictor4':
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        scores = predict_tile_scores(model, gray, tiles, input_size, device, fp_penalty)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        selector_ms = 1000.0 * (time.perf_counter() - t0)
    chosen = choose_tiles(policy, image_name, budget, len(tiles), seed, scores)
    return apply_tiles(base, cubic, tiles, chosen, feather), chosen, selector_ms


def policy_summary(rows: list[dict], fp_penalty: float) -> dict:
    """Calculate ACTUAL combined-region counts, never sum single-tile ROI labels."""
    if not rows:
        raise ValueError('No evaluation rows')
    result = {}
    baseline = next((group for name, group in groupby_policy(rows) if name == 'linear_640'), None)
    if not baseline:
        raise ValueError('Missing linear baseline')
    sums0 = {k: sum(int(r[k]) for r in baseline) for k in ('small_tp', 'all_tp', 'fp', 'small_gt')}
    for policy, group in groupby_policy(rows):
        cnt = {k: sum(int(r[k]) for r in group) for k in ('small_tp', 'all_tp', 'fp', 'small_gt')}
        d_small = cnt['small_tp'] - sums0['small_tp']
        d_fp = cnt['fp'] - sums0['fp']
        result[policy] = {
            'images': len(group), 'small_gt': cnt['small_gt'],
            'small_true_positives': cnt['small_tp'], 'all_true_positives': cnt['all_tp'],
            'false_positives': cnt['fp'],
            'small_recall_conf_025': round(cnt['small_tp'] / cnt['small_gt'], 6) if cnt['small_gt'] else None,
            'delta_small_tp_vs_linear': d_small,
            'delta_fp_vs_linear': d_fp,
            'observed_utility_total_vs_linear': round(d_small - fp_penalty * d_fp, 5),
            'mean_preparation_ms_excluding_io': round(statistics.mean(float(r['prepare_ms']) for r in group), 4),
            'mean_selector_ms_in_preparation': round(statistics.mean(float(r['selector_ms']) for r in group), 4),
            'mean_detector_ms_excluding_io': round(statistics.mean(float(r['detect_ms']) for r in group), 4),
            'mean_pipeline_ms_excluding_io': round(statistics.mean(float(r['pipeline_ms']) for r in group), 4),
            'enhanced_area_fraction': round(statistics.mean(float(r['enhanced_fraction']) for r in group), 4),
        }
    return result


def groupby_policy(rows: list[dict]):
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_policy[r['policy']].append(r)
    return by_policy.items()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=Path('configs/selective_sr_eval.yaml'))
    ap.add_argument('--limit', type=int, default=4, help='Predictor validation images; 0=all')
    ap.add_argument('--out', type=Path, default=Path('reports/utility/selective_smoke_v1'))
    ap.add_argument('--with-map', action='store_true', help='Also compute Ultralytics mAP per policy')
    args = ap.parse_args()
    if args.limit < 0 or not args.config.is_file():
        ap.error('Missing config or invalid --limit')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict) or REQUIRED - set(cfg):
        ap.error(f'Configuration missing: {sorted(REQUIRED - set(cfg if isinstance(cfg,dict) else []))}')
    lr_root = Path(cfg['train_lr'])
    if lr_root.name != 'train' or lr_root.parent.name != 'images' or 'yolo_flir_lr_x2' not in lr_root.parts:
        ap.error('Data-leakage guard: train_lr must be the Step 5A FLIR TRAIN LR directory')
    if not 0 < float(cfg['confidence']) < 1 or int(cfg['imgsz']) != 640:
        ap.error('Must use 640 detector input and confidence in (0,1)')
    if int(cfg['budget_tiles']) != 4 or int(cfg['feather_hr_px']) != 8:
        ap.error('Step 5A compatibility: require 4 tiles, feather=8')
    if int(cfg['val_batch']) not in (1, 2):
        ap.error('CUDA-safety: require val_batch 1 or 2')
    if abs(float(cfg['false_positive_penalty']) - 0.25) > 1e-9:
        ap.error('Must use Step 5A false-positive penalty=0.25')
    for key in ('utility_csv', 'split_manifest', 'predictor_checkpoint', 'detector_checkpoint'):
        if not Path(cfg[key]).is_file():
            ap.error(f'Missing required file: {cfg[key]}')
    if args.out.exists():
        ap.error(f'Experiment directory exists: {args.out}; choose new --out')
    # Keep intermediate arrays and YOLO labels outside report folders.
    dest = Path('data/interim/selective_eval') / args.out.name
    if dest.exists():
        ap.error(f'Intermediate experiment already exists: {dest}; choose new --out')
    paths = get_eval_image_names(Path(cfg['utility_csv']), Path(cfg['split_manifest']), args.limit)
    if not paths:
        ap.error('No eligible predictor-validation images')
    detector_ckpt = Path(cfg['detector_checkpoint'])
    if not torch.cuda.is_available() or str(cfg['device']) != '0':
        ap.error('This pilot requires your CUDA GPU (device 0)')
    device = torch.device('cuda:0')
    model, input_size = predictor_model(Path(cfg['predictor_checkpoint']), Path(cfg['utility_csv']), device)
    from ultralytics import YOLO
    detector = YOLO(str(detector_ckpt))
    predictor_input_size = input_size
    labels_root = lr_root.parent.parent / 'labels' / 'train'
    source_yaml = lr_root.parent.parent / 'data.yaml'
    if not lr_root.is_dir() or not labels_root.is_dir() or not source_yaml.is_file():
        ap.error('Required FLIR LR training images, labels or data.yaml missing')
    tiles = roi_grid(640, 512, 4, 4)
    conf = float(cfg['confidence'])
    if abs(conf - .25) > 1e-9 or abs(float(cfg['nms_iou']) - .7) > 1e-9:
        ap.error('Use the exact Step 5A diagnostic settings (conf=.25, nms_iou=.7)')
    args.out.mkdir(parents=True)
    # Warm both networks before reporting per-image forward timings.
    warm = np.zeros(LR_SHAPE, dtype=np.uint8)
    _ = predict_tile_scores(model, warm, tiles, input_size, device, float(cfg['false_positive_penalty']))
    _ = prediction_records(detector, cv2.resize(warm, HR_SIZE), cfg)
    records = []
    from evaluate_bicubic_baseline import write_data_yaml
    names = yaml.safe_load(source_yaml.read_text(encoding='utf-8'))['names']
    data_files = {}
    for policy in POLICIES:
        folder = dest / policy
        (folder / 'images' / 'val').mkdir(parents=True)
        (folder / 'labels' / 'val').mkdir(parents=True)
        data_files[policy] = write_data_yaml(folder, lr_root.parent.parent, names)
    for i, name in enumerate(paths, 1):
        gray = cv2.imread(str(lr_root / name), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != LR_SHAPE:
            raise ValueError(f'Expected 320x256 LR grayscale: {name}')
        label = labels_root / (Path(name).stem + '.txt')
        if not label.is_file():
            raise FileNotFoundError(f'Missing annotation: {label}')
        gts = ground_truth(label)  # used ONLY after policy selection and YOLO inference
        for policy in POLICIES:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            enhanced, selected, selector_ms = build_policy_image(
                policy, gray, tiles, int(cfg['budget_tiles']), int(cfg['seed']), name,
                int(cfg['feather_hr_px']), model, predictor_input_size, device,
                float(cfg['false_positive_penalty']))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            preds = prediction_records(detector, enhanced, cfg)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            cnt = detection_counts(gts, preds, conf)
            if not cv2.imwrite(str(dest / policy / 'images' / 'val' / name), enhanced):
                raise RuntimeError(f'Cannot write rendered validation image: {policy} {name}')
            import shutil
            shutil.copy2(label, dest / policy / 'labels' / 'val' / label.name)
            area = sum((tiles[k][2] - tiles[k][0]) * (tiles[k][3] - tiles[k][1]) for k in selected)
            records.append({
                'image': name, 'sequence': sequence_id(Path(name)), 'policy': policy,
                'selected_tiles': ' '.join(str(k) for k in selected),
                'enhanced_fraction': area / (640 * 512),
                'small_gt': cnt['small_gt'], 'small_tp': cnt['small_tp'],
                'all_tp': cnt['all_tp'], 'fp': cnt['fp'],
                'prepare_ms': round((t1 - t0) * 1000, 4),
                'selector_ms': round(selector_ms, 4),
                'detect_ms': round((t2 - t1) * 1000, 4),
                'pipeline_ms': round((t2 - t0) * 1000, 4),
            })
        print(f'Compared {i}/{len(paths)} predictor-held-out TRAIN images', flush=True)
    results = policy_summary(records, float(cfg['false_positive_penalty']))
    if args.with_map:
        for policy in POLICIES:
            print(f'Standard YOLO validation AP: {policy}', flush=True)
            metric = detector.val(
                data=str(data_files[policy].resolve()), imgsz=640,
                batch=int(cfg['val_batch']), device='0', plots=False,
                verbose=False, workers=0, save_json=False,
                project=str((args.out / 'yolo_val').resolve()), name=policy,
            )
            results[policy]['map50'] = round(float(metric.box.map50), 6)
            results[policy]['map50_95'] = round(float(metric.box.map), 6)
    with (args.out / 'per_image.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    meta = {
        'task': 'UtiTherm-SR Step 6A — predictor-held-out end-to-end selective ROI pilot',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'images': len(paths), 'policies': list(POLICIES),
        'selection_domain': 'ONLY Step 5B predictor val sequences inside detector TRAIN data',
        'checkpoint_predictor': str(cfg['predictor_checkpoint']),
        'checkpoint_detector': str(detector_ckpt),
        'split_sha256': sha256(Path(cfg['split_manifest'])),
        'roi_measurements_sha256': sha256(Path(cfg['utility_csv'])),
        'config': cfg,
        'results': results,
        'limitations': [
            'TRUE combined-region YOLO TP/FP measured; NOT sums of Step 5A marginal proxies.',
            'Detector already trained on these training-domain frames: NOT independent detector evaluation.',
            'Predictor was not trained on these sequence IDs, but checkpoint selection used this internal validation split.',
            'This is an exploratory internal validation analysis, NOT a final unbiased test.',
            'All 4-tile policies enhance 25% of image AREA, but full-frame bicubic is calculated; NO SR compute saving is demonstrated.',
            'Timing excludes disk I/O and annotation processing; separate GPU passes and image ordering limit fair speed claims.',
            'Full bicubic is OpenCV INTER_CUBIC, unlike the earlier Step 4A PIL interpolation (do NOT compare exact historical metrics).',
            'Ground-truth boxes are used only AFTER selecting regions, for offline evaluation.',
        ],
    }
    (args.out / 'summary.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print('\nACTUAL multi-tile detector diagnostic (small IoU=.5, conf=.25):')
    for policy in POLICIES:
        stat = results[policy]
        print(f"  {policy:18} small recall={stat['small_recall_conf_025']:.3f} "
              f"FP={stat['false_positives']} ΔsmallTP={stat['delta_small_tp_vs_linear']:+} "
              f"ΔFP={stat['delta_fp_vs_linear']:+} "
              f"combined utility={stat['observed_utility_total_vs_linear']:+.2f}", flush=True)
    print(f'DONE — {args.out / "summary.json"} ; {args.out / "per_image.csv"}', flush=True)


if __name__ == '__main__':
    main()
