"""Step 9C: SCREEN whether heavier FSRCNN variants make selective ROI SR faster.

Performance feasibility ONLY. The larger models have random, UNTRAINED weights;
never interpret their images as meaningful thermal super-resolution or detection.
Uses the project's full-frame FSRCNN renderer and batched ROI renderer.

Example (run in project root via Docker):
  python scripts/benchmark_fsrcnn_break_even_step9c.py \
    --out reports/performance/fsrcnn_break_even_step9c_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utitherm.sr.fsrcnn import FSRCNN
from utitherm.sr.roi_fsrcnn import render_fsrcnn
from utitherm.sr.roi_fsrcnn_batched import render_batched_fsrcnn

TILES = [(c * 160, r * 128, (c + 1) * 160, (r + 1) * 128)
         for r in range(4) for c in range(4)]
FIXED_FOUR = [0, 5, 10, 15]  # isolate SR cost; not a quality/selection experiment
DEFAULT_SETTINGS = Path('configs/fsrcnn_break_even_step9c.yaml')
REQUIRED_SETTINGS = {
    'data_config', 'sr_checkpoint', 'frames', 'repeats', 'warmup',
    'seed', 'selector_ms', 'candidates'
}


def load_settings(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Step 9C settings missing: {path}')
    settings = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(settings, dict) or REQUIRED_SETTINGS - set(settings):
        raise ValueError('Step 9C YAML missing required settings')
    for name in ('data_config', 'sr_checkpoint'):
        if not isinstance(settings[name], str) or not settings[name].strip():
            raise ValueError(f'{name} must be a nonempty path')
    for name, minimum in [('frames', 2), ('repeats', 3), ('warmup', 2), ('seed', 0)]:
        value = settings[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    value = settings['selector_ms']
    if type(value) not in (int, float) or not 0 <= value < 1000:
        raise ValueError('selector_ms must be a nonnegative finite-size number')
    raw = settings['candidates']
    if not isinstance(raw, list) or not raw:
        raise ValueError('candidates must be a nonempty list')
    seen = set()
    candidates = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {'name', 'scale', 'd', 's', 'm'}:
            raise ValueError('Candidate needs exactly name, scale, d, s, m')
        name = item['name']
        if not isinstance(name, str) or not name.endswith('_untrained') or name in seen:
            raise ValueError('Candidate name must be unique and end with _untrained')
        seen.add(name)
        kw = {key: item[key] for key in ('scale', 'd', 's', 'm')}
        if any(type(v) is not int for v in kw.values()) or kw['scale'] != 2 or kw['d'] <= 0 or not 0 < kw['s'] <= kw['d'] or kw['m'] < 1:
            raise ValueError(f'Invalid architecture for {name}')
        candidates.append((name, kw))
    settings = dict(settings)
    settings['candidates'] = candidates
    return settings


def break_even_estimate(full_ms: float, roi_ms: float, selector_ms: float) -> dict:
    """This is a timing *estimate*, not an end-to-end detector measurement."""
    if min(full_ms, roi_ms, selector_ms) < 0:
        raise ValueError('Timing inputs cannot be negative')
    estimated_roi_ms = roi_ms + selector_ms
    return {
        'roi_plus_estimated_selector_ms': round(estimated_roi_ms, 5),
        'estimated_full_minus_roi_with_selector_ms': round(full_ms - estimated_roi_ms, 5),
        'potential_break_even': full_ms > estimated_roi_ms,
    }



def load_frames(config_path: Path, n: int, seed: int) -> list[np.ndarray]:
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    root = Path(cfg['train_lr'])
    if not root.is_dir():
        raise FileNotFoundError(f"train_lr missing: {root}")
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if len(paths) < n:
        raise ValueError(f'Need {n} LR images; found {len(paths)} in {root}')
    rng = random.Random(seed)
    chosen = rng.sample(paths, n)
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in chosen]
    for p, arr in zip(chosen, frames):
        if arr is None or arr.shape != (256, 320) or arr.dtype != np.uint8:
            raise ValueError(f'Invalid 320x256 LR thermal frame: {p}')
    return frames


def load_trained_model(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f'Original FSRCNN checkpoint missing: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or 'state_dict' not in payload or 'model_kwargs' not in payload:
        raise ValueError('Expected FSRCNN checkpoint with state_dict and model_kwargs')
    kw = payload['model_kwargs']
    if kw.get('scale') != 2:
        raise ValueError('Expected 2x SR model')
    model = FSRCNN(**kw)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.to(device).eval(), kw


def render(mode, frame, model, device):
    if mode == 'full':
        output, _ = render_fsrcnn(
            frame, TILES, FIXED_FOUR, model, device, 'full', feather=8, halo=16
        )
    elif mode == 'roi4':
        output, _ = render_batched_fsrcnn(
            frame, TILES, FIXED_FOUR, model, device, feather=8, halo=16
        )
    else:
        raise ValueError(mode)
    if output.shape != (512, 640) or output.dtype != np.uint8:
        raise RuntimeError(f'Unexpected {mode} result: {output.shape} {output.dtype}')


def timed_render(mode, frame, model, device) -> float:
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    render(mode, frame, model, device)
    torch.cuda.synchronize(device)
    return (time.perf_counter_ns() - start) / 1e6


def benchmark(label, model, config, frames, device, repeats, warmup, seed, selector_ms):
    rng = random.Random(seed)
    modes = ['full', 'roi4']
    for _ in range(warmup):
        for mode in modes:
            render(mode, frames[0], model, device)
    torch.cuda.synchronize(device)
    timings = {mode: [] for mode in modes}
    for r in range(repeats):
        order = list(range(len(frames)))
        rng.shuffle(order)
        for j in order:
            mode_order = modes[:] if (j + r) % 2 else modes[::-1]
            for mode in mode_order:
                timings[mode].append(timed_render(mode, frames[j], model, device))
    full = statistics.median(timings['full'])
    roi = statistics.median(timings['roi4'])
    # An estimate based on separate Step 8B selector benchmark, NOT a measured pipeline time.
    estimate = break_even_estimate(full, roi, selector_ms)
    estimated_roi_with_selector = roi + selector_ms
    result = {
        'architecture': label,
        'config': config,
        'parameters': sum(p.numel() for p in model.parameters()),
        'samples_per_mode': len(timings['full']),
        'median_full_sr_ms': round(full, 5),
        'median_batched_roi_sr_ms': round(roi, 5),
        'median_sr_saving_ms': round(full - roi, 5),
        'selector_overhead_ms_external_estimate': selector_ms,
        **estimate,
    }
    print(f"{label:20s} params={result['parameters']:>9,} | "
          f"full={full:7.3f} ms | roi4={roi:7.3f} ms | "
          f"SR saved={full - roi:+7.3f} ms | "
          f"after estimated selector={full - estimated_roi_with_selector:+7.3f} ms")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings', type=Path, default=DEFAULT_SETTINGS,
                   help='Dedicated Step 9C YAML (keeps original CLI override options)')
    p.add_argument('--config', type=Path, default=None,
                   help='Override dataset config (original option)')
    p.add_argument('--sr-checkpoint', type=Path, default=None)
    p.add_argument('--frames', type=int, default=None)
    p.add_argument('--repeats', type=int, default=None)
    p.add_argument('--warmup', type=int, default=None)
    p.add_argument('--selector-ms', type=float, default=None,
                   help='External Step 8B selector estimate in ms; not measured here')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    try:
        cfg = load_settings(args.settings)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    data_config = args.config or Path(cfg['data_config'])
    sr_checkpoint = args.sr_checkpoint or Path(cfg['sr_checkpoint'])
    frames_n = args.frames if args.frames is not None else cfg['frames']
    repeats = args.repeats if args.repeats is not None else cfg['repeats']
    warmup = args.warmup if args.warmup is not None else cfg['warmup']
    selector_ms = args.selector_ms if args.selector_ms is not None else cfg['selector_ms']
    seed = args.seed if args.seed is not None else cfg['seed']
    if frames_n < 2 or repeats < 3 or warmup < 2 or seed < 0 or not 0 <= selector_ms < 1000:
        p.error('Require frames>=2, repeats>=3, warmup>=2, seed>=0, selector-ms>=0')
    if args.out.exists():
        p.error(f'Output exists; use a new path: {args.out}')
    if not torch.cuda.is_available():
        p.error('NVIDIA CUDA device required for representative benchmarking')
    torch.manual_seed(seed)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    frames = load_frames(data_config, frames_n, seed)
    original, kw = load_trained_model(sr_checkpoint, device)
    results = [benchmark('trained_original', original, kw, frames, device,
                         repeats, warmup, seed, selector_ms)]
    del original
    torch.cuda.empty_cache()
    for label, kwargs in cfg['candidates']:
        model = FSRCNN(**kwargs).to(device).eval()
        results.append(benchmark(label, model, kwargs, frames, device,
                                 repeats, warmup, seed, selector_ms))
        del model
        torch.cuda.empty_cache()
    report = {
        'task': 'Step 9C untrained-model computational feasibility screening',
        'utc': datetime.now(timezone.utc).isoformat(),
        'gpu': torch.cuda.get_device_name(0),
        'settings': str(args.settings),
        'settings_sha256': hashlib.sha256(args.settings.read_bytes()).hexdigest(),
        'data_config': str(data_config),
        'trained_checkpoint': str(sr_checkpoint),
        'trained_checkpoint_sha256': hashlib.sha256(sr_checkpoint.read_bytes()).hexdigest(),
        'frame_count': len(frames),
        'repeats': repeats,
        'warmup': warmup,
        'seed': seed,
        'selected_tiles': FIXED_FOUR,
        'halo': 16,
        'full_lr_pixels': 81920,
        'batched_roi_lr_pixels': 43008,
        'models': results,
        'limitations': [
            'Larger SR networks are RANDOMLY INITIALIZED; no image quality or detection conclusions.',
            'Selection is fixed, not predictor selected; detector is excluded from timing.',
            'Selector time is estimated from a DIFFERENT benchmark; threshold not same-run proven.',
            'Increasing mapping depth expands receptive field; halo adequacy must be retested.',
            'Small single-GPU performance screen only; rerun interleaved full pipeline if promising.',
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'SAVED — {args.out}')


if __name__ == '__main__':
    main()
