"""Step 9F3: compare selective Large FSRCNN ROI output with TRUE full-frame SR.

Usage:
  python scripts/check_large_fsrcnn_fullframe_equiv_step9f.py \
    --config configs/large_fsrcnn_fullframe_equiv_step9f.yaml \
    --out reports/performance/large_fsrcnn_equiv_smoke_step9f_v1.json

No models are changed and no YOLO metrics are computed. Reports are non-overwriting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compare_tile(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Compare uint8 SR tile arrays, not blended background pixels."""
    if reference.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError('Require two uint8 SR tiles')
    if reference.ndim != 2 or candidate.shape != reference.shape or not reference.size:
        raise ValueError('Require matching, nonempty 2D tile arrays')
    diff = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
    return {
        'max_difference': int(diff.max()),
        'mae': float(diff.mean()),
        'different_pixels': int(np.count_nonzero(diff)),
        'num_pixels': int(diff.size),
    }


def summarize_rows(rows: list[dict], halos: list[int], tolerance: int) -> dict:
    if not rows or tolerance < 0:
        raise ValueError('No comparison rows or invalid tolerance')
    summary = {}
    for halo in halos:
        sub = [r for r in rows if r['halo'] == halo]
        if not sub:
            raise ValueError(f'No comparisons for halo={halo}')
        bad = [r for r in sub if r['max_difference'] > tolerance]
        summary[str(halo)] = {
            'images': len(sub),
            'max_abs_pixel_difference_vs_full': max(r['max_difference'] for r in sub),
            'mean_absolute_difference_vs_full': sum(r['abs_error_total'] for r in sub) / sum(r['num_pixels'] for r in sub),
            'fraction_pixels_changed_vs_full': sum(r['different_pixels'] for r in sub) / sum(r['num_pixels'] for r in sub),
            'images_exceeding_tolerance': len(bad),
            'within_tolerance_on_all_images': not bad,
        }
    return summary


def load_config(path: Path) -> dict:
    import yaml
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    required = {'selective_config', 'sr_checkpoint', 'predictor_checkpoint',
                'predictor_source_csv', 'halos', 'limit', 'max_uint8_difference_tolerance'}
    if not isinstance(cfg, dict) or required - set(cfg):
        raise ValueError(f'Missing config keys: {sorted(required - set(cfg or {}))}')
    halos = cfg['halos']
    if not isinstance(halos, list) or not halos or any(type(h) is not int or h < 10 for h in halos) or len(set(halos)) != len(halos):
        raise ValueError('Require unique integer halos >=10')
    if type(cfg['limit']) is not int or cfg['limit'] < 0:
        raise ValueError('Require limit>=0')
    if type(cfg['max_uint8_difference_tolerance']) is not int or not 0 <= cfg['max_uint8_difference_tolerance'] <= 255:
        raise ValueError('Invalid tolerance')
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/large_fsrcnn_fullframe_equiv_step9f.yaml'))
    p.add_argument('--limit', type=int, default=None, help='Override config; 0 = all available frames')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not a.config.is_file():
        p.error(f'Config missing: {a.config}')
    if a.out.exists():
        p.error(f'Refusing to overwrite: {a.out}')
    cfg = load_config(a.config)
    limit = cfg['limit'] if a.limit is None else a.limit
    if limit < 0:
        p.error('Limit must be >=0')

    # Import runtime dependencies only after CLI/config validation.
    import cv2
    import torch
    import yaml
    sys.path.insert(0, str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from benchmark_selective_fsrcnn import load_sr_checkpoint
    from evaluate_selective_sr import (
        choose_tiles, get_eval_image_names, predict_tile_scores, predictor_model)
    from probe_roi_utility import roi_grid
    from utitherm.sr.roi_fsrcnn_batched import batched_fsrcnn_tiles

    if not torch.cuda.is_available():
        p.error('CUDA GPU required for exact full-frame vs ROI model comparison')
    paths = {k: Path(cfg[k]) for k in (
        'selective_config', 'sr_checkpoint', 'predictor_checkpoint', 'predictor_source_csv')}
    if any(not file.is_file() for file in paths.values()):
        p.error(f'Missing file: {[str(f) for f in paths.values() if not f.is_file()]}')
    select_cfg = yaml.safe_load(paths['selective_config'].read_text(encoding='utf-8'))
    lr_root = Path(select_cfg['train_lr'])
    label_csv = Path(select_cfg['utility_csv'])
    label_manifest = Path(select_cfg['split_manifest'])
    names = get_eval_image_names(label_csv, label_manifest, limit=limit)
    if not names:
        p.error('No selected images')
    device = torch.device('cuda:0')
    cv2.setNumThreads(1)
    sr = load_sr_checkpoint(paths['sr_checkpoint'], device)
    selector, input_size = predictor_model(
        paths['predictor_checkpoint'], paths['predictor_source_csv'], device)
    tiles = roi_grid(640, 512, 4, 4)
    rows = []
    for j, name in enumerate(names, 1):
        gray = cv2.imread(str(lr_root / name), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.dtype != np.uint8 or gray.shape != (256, 320):
            raise ValueError(f'Invalid source thermal image: {name}')
        scores = predict_tile_scores(selector, gray, tiles, input_size, device,
                                     float(select_cfg['false_positive_penalty']))
        chosen = choose_tiles('predictor4', name, 4, len(tiles), int(select_cfg['seed']), scores)
        if len(chosen) != 4:
            raise ValueError(f'Expected 4 tiles, got {chosen}')
        # Exactly the same input normalization and uint8 rounding as the batched ROI renderer.
        x = torch.from_numpy(gray.astype(np.float32)[None, None]).to(device).div_(255.0)
        with torch.inference_mode():
            y = sr(x)
            if tuple(y.shape) != (1, 1, 512, 640) or not bool(torch.isfinite(y).all()):
                raise ValueError('Invalid full-frame model output')
            full = y.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()[0, 0]
        for halo in cfg['halos']:
            patches, input_pixels = batched_fsrcnn_tiles(gray, tiles, chosen, sr, device, halo=halo)
            accum = {'max_difference': 0, 'different_pixels': 0,
                     'num_pixels': 0, 'abs_error_total': 0.0}
            for i in chosen:
                x0, y0, x1, y1 = tiles[i]
                diff = compare_tile(full[y0:y1, x0:x1], patches[i])
                accum['max_difference'] = max(accum['max_difference'], diff['max_difference'])
                accum['different_pixels'] += diff['different_pixels']
                accum['num_pixels'] += diff['num_pixels']
                accum['abs_error_total'] += diff['mae'] * diff['num_pixels']
            rows.append({'image': name, 'halo': halo, 'selected': chosen,
                         'model_input_lr_pixels': input_pixels, **accum})
        if j == 1 or j % 5 == 0 or j == len(names):
            print(f'Compared {j}/{len(names)} images vs full-frame Large FSRCNN', flush=True)

    summary = summarize_rows(rows, cfg['halos'], cfg['max_uint8_difference_tolerance'])
    report = {
        'task': 'Step 9F3 full-frame Large FSRCNN vs batched selected tile pixel audit',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': str(a.config), 'config_sha256': sha256_file(a.config),
        'num_images': len(names), 'images_source': 'V1 predictor-held-out TRAIN-domain images',
        'threshold_max_abs_pixel_difference': cfg['max_uint8_difference_tolerance'],
        'files': {key: {'path': str(path), 'sha256': sha256_file(path)} for key, path in paths.items()},
        'results': summary, 'per_image': rows,
        'limitations': [
            'Compared raw enhanced ROI pixels to full-frame output, before blending.',
            'No detector was run; near-equal intensities may still change detection.',
            'No independent final test: training-domain engineering images only.',
            'This test measures correspondence, not speed; see Step 9B/9F2 timing.'
        ]
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\nFULL-FRAME vs SELECTIVE LARGE FSRCNN (raw tiles):')
    for halo in cfg['halos']:
        r = summary[str(halo)]
        status = 'WITHIN TOLERANCE' if r['within_tolerance_on_all_images'] else 'EXCEEDS TOLERANCE'
        print(f"halo={halo:2d} | max diff={r['max_abs_pixel_difference_vs_full']} | "
              f"changed={r['fraction_pixels_changed_vs_full']:.7f} | "
              f"MAE={r['mean_absolute_difference_vs_full']:.7f} | {status}")
    print(f'SAVED — {a.out}')


if __name__ == '__main__':
    main()
