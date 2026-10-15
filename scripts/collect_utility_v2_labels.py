"""Collect NEW Step 7B2A counterfactual ROI labels from a frozen TRAIN manifest.

Resumable by image, with atomic per-image JSON shards to avoid redoing thousands
of inference calls if a container is interrupted. Does NOT retrain detector.
Requires original Step 5A scripts/probe_roi_utility.py in existing project.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from prepare_utility_v2_manifest import verify_manifest, sha256, sequence_id
from probe_roi_utility import (detection_counts, enhanced_tile, prediction_records,
                               roi_grid, utility)
from evaluate_bicubic_baseline import ground_truth

COLUMNS = ('image', 'sequence', 'tile_index', 'tile_x0_hr', 'tile_y0_hr', 'tile_x1_hr',
           'tile_y1_hr', 'enhanced_fraction', 'baseline_small_tp', 'baseline_fp',
           'candidate_small_tp', 'candidate_fp', 'delta_small_tp', 'delta_all_tp',
           'delta_fp', 'utility_score', 'positive_label')


def shard_is_valid(item: dict, image_name: str, ntiles: int) -> bool:
    try:
        records = item['rows']
        return (item['image'] == image_name and isinstance(records, list) and
                len(records) == ntiles and
                {int(r['tile_index']) for r in records} == set(range(ntiles)) and
                all(r['image'] == image_name and r['sequence'] == sequence_id(image_name)
                    for r in records) and
                all(set(COLUMNS) <= set(r) for r in records))
    except (ValueError, KeyError, TypeError):
        return False


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=Path('configs/utility_v2_data.yaml'))
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=6, help='First N manifest images; 0=entire frozen manifest')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resume', action='store_true', help='Resume an interrupted, matching run')
    args = ap.parse_args()
    if args.limit < 0:
        ap.error('--limit must be non-negative')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    frozen = json.loads(args.manifest.read_text(encoding='utf-8'))
    all_images = verify_manifest(frozen, cfg)
    images = all_images[:args.limit] if args.limit else all_images
    if not images:
        ap.error('No frozen images to process')
    if (int(cfg['grid_rows']), int(cfg['grid_cols'])) != (4, 4):
        ap.error('Existing 16-ROI predictor requires grid 4x4')
    if int(cfg['imgsz']) != 640:
        ap.error('Experiment requires YOLO imgsz=640')
    expected = dict(protocol='STEP_7B2A_RESUMABLE_TRAIN_ROI_LABELS',
                    selected_images=[p.name for p in images],
                    frozen_manifest_sha256=sha256(args.manifest),
                    config_sha256=sha256(args.config),
                    checkpoint_sha256=sha256(Path(cfg['detector_checkpoint'])),
                    num_tiles=16)
    run_file = args.out / 'run_manifest.json'
    if args.resume:
        if not run_file.is_file() or json.loads(run_file.read_text(encoding='utf-8')) != expected:
            raise ValueError('Refusing unsafe resume: manifest, config, or checkpoint differs')
    else:
        if args.out.exists():
            ap.error('Output already exists; choose another --out or use --resume for same experiment')
        (args.out / 'shards').mkdir(parents=True)
        run_file.write_text(json.dumps(expected, indent=2)+'\n', encoding='utf-8')
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this research run')
    detector = YOLO(str(cfg['detector_checkpoint']))
    tiles = roi_grid(640, 512, 4, 4)
    threshold = float(cfg['confidence'])
    if not 0 < threshold < 1:
        raise ValueError('Confidence outside (0,1)')
    lab_dir = Path(cfg['train_lr']).parent.parent/'labels'/'train'
    all_shards = []
    for index, imgpath in enumerate(images, 1):
        shard_path = args.out/'shards'/f'{imgpath.stem}.json'
        if shard_path.exists():
            item = json.loads(shard_path.read_text(encoding='utf-8'))
            if not shard_is_valid(item, imgpath.name, 16):
                raise ValueError(f'Incomplete or corrupted progress shard: {shard_path}')
        else:
            gray = cv2.imread(str(imgpath), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (256, 320):
                raise ValueError(f'Expected grayscale 320x256: {imgpath}')
            gts = ground_truth(lab_dir/f'{imgpath.stem}.txt')
            linear = cv2.resize(gray, (640, 512), interpolation=cv2.INTER_LINEAR)
            bicubic = cv2.resize(gray, (640, 512), interpolation=cv2.INTER_CUBIC)
            baseline = detection_counts(gts, prediction_records(detector, linear, cfg), threshold)
            rows = []
            for tile_index, rect in enumerate(tiles):
                intervention = enhanced_tile(linear, bicubic, rect, int(cfg['feather_hr_px']))
                result = detection_counts(gts, prediction_records(detector, intervention, cfg), threshold)
                delta = utility(baseline, result, float(cfg['false_positive_penalty']))
                rows.append(dict(image=imgpath.name, sequence=sequence_id(imgpath.name),
                                 tile_index=tile_index, tile_x0_hr=rect[0], tile_y0_hr=rect[1],
                                 tile_x1_hr=rect[2], tile_y1_hr=rect[3], enhanced_fraction=0.0625,
                                 baseline_small_tp=baseline['small_tp'], baseline_fp=baseline['fp'],
                                 candidate_small_tp=result['small_tp'], candidate_fp=result['fp'], **delta))
            item = dict(image=imgpath.name, baseline=baseline, rows=rows)
            tmp_path = shard_path.with_suffix('.json.tmp')
            tmp_path.write_text(json.dumps(item, separators=(',', ':')), encoding='utf-8')
            os.replace(tmp_path, shard_path)
        all_shards.append(item)
        if index == 1 or index % 10 == 0 or index == len(images):
            print(f'Measured {index}/{len(images)} fresh TRAIN images; {16*index} ROI interventions', flush=True)
    rows = [r for shard in all_shards for r in shard['rows']]
    baselines = [dict(image=s['image'], sequence=sequence_id(s['image']), **s['baseline']) for s in all_shards]
    write_csv(args.out/'roi_utility.csv', rows, COLUMNS)
    write_csv(args.out/'baseline_per_image.csv', baselines,
              ('image', 'sequence', 'small_tp', 'all_tp', 'fp', 'small_gt'))
    result = dict(task='Step 7B2A fresh detector-TRAIN-domain utility labels',
                  created_utc=datetime.now(timezone.utc).isoformat(), images=len(images),
                  unique_sequences=len({sequence_id(p.name) for p in images}), interventions=len(rows),
                  positive_utility=sum(float(r['utility_score'])>0 for r in rows),
                  small_gain_interventions=sum(int(r['delta_small_tp'])>0 for r in rows),
                  small_loss_interventions=sum(int(r['delta_small_tp'])<0 for r in rows),
                  v1_sequences_excluded=True,
                  note='Training-domain counterfactual targets from an already trained detector; not independent test evidence or a speed claim.')
    (args.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(f'DONE — {args.out/"roi_utility.csv"} | {len(rows)} interventions | '
          f'positive={result["positive_utility"]}; small gains={result["small_gain_interventions"]}; '
          f'small losses={result["small_loss_interventions"]}', flush=True)


if __name__ == '__main__':
    main()
