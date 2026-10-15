"""Step 11A: Large FSRCNN single-tile incremental utility over FULL Bicubic.

No model training. Operates only on locked Step 7B2A detector-TRAIN images.
From the project root inside research Docker:
  python scripts/collect_bicubic_first_feasibility_step11a.py --limit 4 --out reports/utility/step11a_smoke_4_v1
  python scripts/collect_bicubic_first_feasibility_step11a.py --limit 40 --out reports/utility/step11a_pilot_40_v1
  python scripts/collect_bicubic_first_feasibility_step11a.py --limit 0 --out reports/utility/step11a_full_300_v1
  python scripts/collect_bicubic_first_feasibility_step11a.py --limit 0 --resume --out reports/utility/step11a_full_300_v1

WARNING: Detector trained on these frames; NOT independent evidence. 16
single-tile effects CANNOT simply be added to predict four-tile outcomes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_utility_v2_manifest import verify_manifest, sha256, sequence_id  # noqa: E402
from probe_roi_utility import roi_grid, prediction_records, detection_counts, utility  # noqa: E402
from evaluate_bicubic_baseline import ground_truth  # noqa: E402
from benchmark_selective_fsrcnn import load_sr_checkpoint  # noqa: E402
from utitherm.sr.bicubic_first_fsrcnn_step11a import render_bicubic_first_fsrcnn  # noqa: E402

SCHEMA = 'STEP_11A_INCREMENTAL_LARGE_FSRCNN_OVER_BICUBIC_SINGLE_ROI_V1'
COLUMNS = (
    'image', 'sequence', 'tile_index', 'tile_x0_hr', 'tile_y0_hr',
    'tile_x1_hr', 'tile_y1_hr', 'baseline_kind', 'processed_lr_pixels',
    'baseline_small_tp', 'baseline_all_tp', 'baseline_fp',
    'candidate_small_tp', 'candidate_all_tp', 'candidate_fp',
    'delta_small_tp', 'delta_all_tp', 'delta_fp', 'utility_score', 'positive_label',
)


def validate_config(exp: dict, base: dict, frozen: dict) -> None:
    if not all(isinstance(x, dict) for x in (exp, base, frozen)):
        raise ValueError('Configuration and manifest must be YAML/JSON mapping objects')
    if (exp.get('baseline') != 'full_bicubic_640' or
            exp.get('intervention') != 'bicubic_base_single_large_fsrcnn_roi_x2_h10_feather8' or
            int(exp.get('halo', -1)) != 10 or
            int(exp.get('feather_hr_px', -1)) != 8 or
            int(exp.get('budget_tiles', -1)) != 1):
        raise ValueError('Step 11A intervention protocol changed')
    if (int(base.get('grid_rows', -1)), int(base.get('grid_cols', -1))) != (4, 4):
        raise ValueError('Need 4x4 fixed ROI grid')
    expected = {'imgsz': 640, 'confidence': 0.25, 'nms_iou': 0.7,
                'false_positive_penalty': 0.25, 'feather_hr_px': 8}
    for k, v in expected.items():
        if abs(float(base.get(k, -999)) - v) > 1e-9:
            raise ValueError(f'Frozen detection/intervention config differs: {k}')
    if str(base.get('device')) != '0':
        raise ValueError('Step 11A GPU protocol requires device 0')
    if (int(frozen.get('num_images', -1)) != int(exp.get('expected_images', -2)) or
            int(frozen.get('unique_sequences', -1)) != int(exp.get('expected_sequences', -2))):
        raise ValueError('Frozen Step 7B2A image and sequence counts differ')


def validate_sr_checkpoint(file: Path) -> dict:
    if not file.is_file():
        raise FileNotFoundError(file)
    state = torch.load(file, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or 'state_dict' not in state:
        raise ValueError('SR checkpoint has no state_dict')
    kw = state.get('model_kwargs')
    if not isinstance(kw, dict) or any(int(kw.get(k, -1)) != v
            for k, v in {'scale': 2, 'd': 168, 's': 36, 'm': 10}.items()):
        raise ValueError('Unexpected SR architecture (need trained Large FSRCNN)')
    return {'epoch': state.get('epoch'), 'model_kwargs': kw}


def provenance(config: Path, base: Path, frozen: Path, detector: Path, sr: Path,
               images: list[str], exp: dict) -> dict:
    return {
        'protocol': SCHEMA, 'selected_images': images,
        'configuration_sha256': sha256(config),
        'base_configuration_sha256': sha256(base),
        'frozen_manifest_sha256': sha256(frozen),
        'detector_checkpoint_sha256': sha256(detector),
        'sr_checkpoint_sha256': sha256(sr),
        'baseline': exp['baseline'], 'intervention': exp['intervention'],
        'halo': 10, 'feather': 8, 'num_tiles': 16,
    }


def shard_is_valid(item: dict, image_name: str) -> bool:
    try:
        if item['image'] != image_name or item['sequence'] != sequence_id(image_name):
            return False
        base = item['baseline']
        if set(('small_tp', 'all_tp', 'fp', 'small_gt')) - set(base):
            return False
        rows = item['rows']
        if len(rows) != 16 or {int(r['tile_index']) for r in rows} != set(range(16)):
            return False
        tiles = roi_grid(640, 512, 4, 4)
        for r in rows:
            if set(COLUMNS) - set(r):
                return False
            i = int(r['tile_index'])
            rect = tiles[i]
            if (r['image'] != image_name or r['sequence'] != item['sequence'] or
                    r['baseline_kind'] != 'full_bicubic_640' or
                    tuple(int(r[k]) for k in ('tile_x0_hr','tile_y0_hr','tile_x1_hr','tile_y1_hr')) != tuple(rect) or
                    int(r['processed_lr_pixels']) != 8400):
                # 80x64 tile with halo 10 -> (100*84)=8400 LR px.
                return False
            if (int(r['baseline_small_tp']) != int(base['small_tp']) or
                    int(r['baseline_all_tp']) != int(base['all_tp']) or
                    int(r['baseline_fp']) != int(base['fp'])):
                return False
            dtp = int(r['candidate_small_tp']) - int(base['small_tp'])
            dall = int(r['candidate_all_tp']) - int(base['all_tp'])
            dfp = int(r['candidate_fp']) - int(base['fp'])
            score = dtp - .25 * dfp
            if (int(r['delta_small_tp']) != dtp or int(r['delta_all_tp']) != dall or
                    int(r['delta_fp']) != dfp or
                    abs(float(r['utility_score']) - score) > 1e-6 or
                    int(r['positive_label']) != int(score > 0)):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return False


def make_rows(gray: np.ndarray, image_name: str, gts: list[dict],
              detector, sr, device, cfg: dict,
              tiles: list[tuple[int, int, int, int]], halo: int = 10) -> tuple[dict, list[dict]]:
    if gray.shape != (256, 320) or gray.dtype != np.uint8 or gray.ndim != 2 or len(tiles) != 16:
        raise ValueError('Require 320x256 grayscale uint8 and 4x4 tiles')
    if halo != 10:
        raise ValueError('Step 11A freezes halo 10')
    cubic = cv2.resize(gray, (640, 512), interpolation=cv2.INTER_CUBIC)
    threshold = float(cfg['confidence'])
    baseline = detection_counts(gts, prediction_records(detector, cubic, cfg), threshold)
    rows = []
    for idx, rect in enumerate(tiles):
        candidate, processed = render_bicubic_first_fsrcnn(
            gray, tiles, [idx], sr, device, feather=8, halo=halo, bicubic=cubic)
        found = detection_counts(gts, prediction_records(detector, candidate, cfg), threshold)
        change = utility(baseline, found, float(cfg['false_positive_penalty']))
        rows.append({
            'image': image_name, 'sequence': sequence_id(image_name),
            'tile_index': idx, 'tile_x0_hr': rect[0], 'tile_y0_hr': rect[1],
            'tile_x1_hr': rect[2], 'tile_y1_hr': rect[3],
            'baseline_kind': 'full_bicubic_640', 'processed_lr_pixels': processed,
            'baseline_small_tp': baseline['small_tp'], 'baseline_all_tp': baseline['all_tp'],
            'baseline_fp': baseline['fp'], 'candidate_small_tp': found['small_tp'],
            'candidate_all_tp': found['all_tp'], 'candidate_fp': found['fp'],
            **change,
        })
    return baseline, rows


def save_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def summarize(rows: list[dict], saved: list[dict], out: Path, record: dict) -> dict:
    positives = [r for r in rows if float(r['utility_score']) > 0]
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r['image'], []).append(r)
    # An ORACLE on isolated ROIs, not an attainable selection policy nor a
    # mathematical upper bound on combined four-ROI performance.
    top4_proxy = [sum(sorted((float(r['utility_score']) for r in rs), reverse=True)[:4])
                  for rs in grouped.values()]
    return {
        'task': 'Step 11A: incremental Large-FSRCNN ROI utility over FULL Bicubic',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': SCHEMA, 'images': len(saved),
        'unique_sequences': len({s['sequence'] for s in saved}),
        'interventions': len(rows), 'baseline': 'full_bicubic_640',
        'positive_utility': len(positives),
        'positive_fraction': round(len(positives) / len(rows), 6),
        'small_gain': sum(int(r['delta_small_tp']) > 0 for r in rows),
        'small_loss': sum(int(r['delta_small_tp']) < 0 for r in rows),
        'fp_reduction': sum(int(r['delta_fp']) < 0 for r in rows),
        'fp_increase': sum(int(r['delta_fp']) > 0 for r in rows),
        'images_with_any_positive': sum(any(float(r['utility_score']) > 0 for r in rs)
                                       for rs in grouped.values()),
        'mean_top4_single_roi_oracle_proxy': round(sum(top4_proxy) / len(top4_proxy), 6),
        'positive_utility_mean': (round(sum(float(r['utility_score']) for r in positives)/len(positives), 6)
                                  if positives else None),
        'bicubic_baseline_small_tp': sum(int(s['baseline']['small_tp']) for s in saved),
        'bicubic_baseline_fp': sum(int(s['baseline']['fp']) for s in saved),
        'source_provenance': record,
        'roi_csv_sha256': sha256(out / 'incremental_roi_utility.csv'),
        'limitations': [
            'Only detector TRAIN images; detector is not out-of-fold.',
            'Large FSRCNN was also trained on FLIR TRAIN data.',
            'The 300-image sample was used for earlier development; exploratory feasibility only.',
            'Isolated single-tile utility is NOT a four-tile combined detection result.',
            'Oracle top-four marginal sum is a nonrealizable ranking proxy, not a guaranteed combined upper bound.',
            'No timing benchmark or inference-time policy; 16 GPU passes plus 17 YOLO detections per frame.',
            'No official mAP or untouched final test; no novelty claim.',
        ],
    }


def run(args: argparse.Namespace) -> None:
    if args.limit < 0:
        raise ValueError('--limit must be nonnegative')
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    exp = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    base_file, frozen_file = Path(exp['base_collection_config']), Path(exp['frozen_manifest'])
    base = yaml.safe_load(base_file.read_text(encoding='utf-8'))
    frozen = json.loads(frozen_file.read_text(encoding='utf-8'))
    validate_config(exp, base, frozen)
    paths = verify_manifest(frozen, base)
    images = paths[:args.limit] if args.limit else paths
    if not images:
        raise ValueError('Frozen manifest contained no images')
    detector_file = Path(base['detector_checkpoint'])
    sr_file = Path(exp['large_sr_checkpoint'])
    sr_info = validate_sr_checkpoint(sr_file)
    record = provenance(args.config, base_file, frozen_file, detector_file,
                        sr_file, [p.name for p in images], exp)
    manifest_out = args.out / 'run_manifest.json'
    if args.resume:
        if not manifest_out.is_file() or json.loads(manifest_out.read_text(encoding='utf-8')) != record:
            raise ValueError('Unsafe resume: run manifest or source hashes differ')
    elif args.out.exists():
        raise FileExistsError(f'Output exists: {args.out}; use --resume if matching')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device 0 required for official Step 11A pilot')
    if not args.resume:
        (args.out / 'shards').mkdir(parents=True)
        manifest_out.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    device = torch.device('cuda:0')
    sr = load_sr_checkpoint(sr_file, device)
    from ultralytics import YOLO
    detector = YOLO(str(detector_file))
    tiles = roi_grid(640, 512, 4, 4)
    labels = Path(base['train_lr']).parent.parent / 'labels' / 'train'
    cv2.setNumThreads(1)
    print(f'Step 11A Bicubic-first: {len(images)} detector-TRAIN frames; '
          f'{len({sequence_id(p.name) for p in images})} sequences; '
          f'16 single-ROI interventions/image; Large SR epoch={sr_info["epoch"]}', flush=True)
    saved = []
    for pos, image in enumerate(images, 1):
        shard_file = args.out / 'shards' / f'{image.stem}.json'
        if shard_file.exists():
            shard = json.loads(shard_file.read_text(encoding='utf-8'))
            if not shard_is_valid(shard, image.name):
                raise ValueError(f'Invalid or corrupted shard; refusing skip: {shard_file}')
        else:
            gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (256, 320) or gray.dtype != np.uint8:
                raise ValueError(f'Invalid 320x256 thermal input: {image}')
            gts = ground_truth(labels / f'{image.stem}.txt')
            baseline, rows = make_rows(gray, image.name, gts, detector, sr,
                                       device, base, tiles, halo=10)
            shard = {'image': image.name, 'sequence': sequence_id(image.name),
                     'baseline': baseline, 'rows': rows}
            if not shard_is_valid(shard, image.name):
                raise RuntimeError(f'Internal shard validation failed: {image.name}')
            tmp = shard_file.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(shard, separators=(',', ':')), encoding='utf-8')
            os.replace(tmp, shard_file)
        saved.append(shard)
        if pos == 1 or pos % 10 == 0 or pos == len(images):
            print(f'Analyzed {pos}/{len(images)} frames; {pos*16} Bicubic-first ROI interventions', flush=True)
    rows = [row for shard in saved for row in shard['rows']]
    baselines = [{'image': shard['image'], 'sequence': shard['sequence'],
                  **shard['baseline']} for shard in saved]
    save_csv(args.out / 'incremental_roi_utility.csv', rows, COLUMNS)
    save_csv(args.out / 'bicubic_baseline_per_image.csv', baselines,
             ('image','sequence','small_tp','all_tp','fp','small_gt'))
    summary = summarize(rows, saved, args.out, record)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(f'DONE — {args.out / "summary.json"} | positive={summary["positive_utility"]}/{len(rows)} '
          f'({100*summary["positive_fraction"]:.1f}%); frames_with_positive='
          f'{summary["images_with_any_positive"]}/{len(images)}; gains={summary["small_gain"]}, '
          f'losses={summary["small_loss"]}; ORACLE marginal top4 proxy='
          f'{summary["mean_top4_single_roi_oracle_proxy"]:.3f}/frame (NOT combined-ROI)', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=Path('configs/bicubic_first_feasibility_step11a.yaml'))
    ap.add_argument('--limit', type=int, default=4, help='0=all 300 frozen images')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resume', action='store_true', help='Resume only a matching existing run')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
