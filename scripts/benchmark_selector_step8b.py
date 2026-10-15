"""Step 8B: safely benchmark a lower-overhead UtiTherm-SR tile selector.

No project source files or checkpoints are modified. Compares the exact Step 6
encoder path against an optimized path using the *same* existing V2 weights,
checks encoded tensors and selected tiles, then benchmarks synchronized runtime.

Run at project root:
  docker compose run --rm research python scripts/benchmark_selector_step8b.py \
    --out reports/performance/selector_step8b_v1.json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utitherm.utility.predictor import (  # noqa: E402
    HEADS, UtilityPredictor, encode_tile, ranking_score,
)

HR = (640, 512)
TILES = [(c * 160, r * 128, (c + 1) * 160, (r + 1) * 128)
         for r in range(4) for c in range(4)]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_encode(gray, tiles, input_size):
    """Exact preprocessing from evaluate_selective_sr.predict_tile_scores."""
    encoded = [encode_tile(gray, rect, input_size=input_size, hr_size=HR)
               for rect in tiles]
    return torch.stack([a for a, _ in encoded]), torch.stack([b for _, b in encoded])


def fast_encode(gray, tiles, input_size):
    """One full-context resize/frame, one NumPy allocation, fixed-shape layout."""
    if gray.ndim != 2 or gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError('Expected nonempty uint8 thermal frame')
    iw, ih = input_size
    hw, hh = HR
    if min(iw, ih) < 1:
        raise ValueError('Invalid input size')
    h, w = gray.shape
    context = cv2.resize(gray, (iw, ih), interpolation=cv2.INTER_AREA)
    packed = np.empty((len(tiles), 2, ih, iw), dtype=np.uint8)
    coords = np.empty((len(tiles), 4), dtype=np.float32)
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        if not (0 <= x0 < x1 <= hw and 0 <= y0 < y1 <= hh):
            raise ValueError('Invalid HR tile rectangle')
        xa = max(0, min(w - 1, round(x0 * w / hw)))
        xb = max(xa + 1, min(w, round(x1 * w / hw)))
        ya = max(0, min(h - 1, round(y0 * h / hh)))
        yb = max(ya + 1, min(h, round(y1 * h / hh)))
        packed[i, 0] = cv2.resize(gray[ya:yb, xa:xb], (iw, ih), interpolation=cv2.INTER_AREA)
        packed[i, 1] = context
        coords[i] = ((x0 + x1) / (2 * hw), (y0 + y1) / (2 * hh),
                     (x1 - x0) / hw, (y1 - y0) / hh)
    return torch.from_numpy(packed.astype(np.float32) / 255.0), torch.from_numpy(coords)


def old_scores(model, gray, tiles, input_size, device, fp_penalty):
    x, coords = reference_encode(gray, tiles, input_size)
    with torch.inference_mode():
        p = model(x.to(device), coords.to(device)).sigmoid().cpu().numpy()
    return np.asarray(ranking_score(p, fp_penalty), dtype=np.float64)


class FastScorer:
    def __init__(self, model, tiles, input_size, device, fp_penalty):
        self.model = model
        self.tiles = tiles
        self.input_size = input_size
        self.device = device
        self.fp_penalty = fp_penalty
        # Fixed grid geometry: calculated once, not per frame.
        _, coords = fast_encode(np.zeros((256, 320), dtype=np.uint8), tiles, input_size)
        self.coords = coords.to(device)

    def __call__(self, gray):
        x, _ = fast_encode(gray, self.tiles, self.input_size)
        with torch.inference_mode():
            p = self.model(x.to(self.device), self.coords).sigmoid().cpu().numpy()
        return np.asarray(ranking_score(p, self.fp_penalty), dtype=np.float64)


def top_four(scores):
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:4]


def synchronized(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(fn, device):
    synchronized(device)
    start = time.perf_counter()
    fn()
    synchronized(device)
    return 1000.0 * (time.perf_counter() - start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('models/utility/roi_utility_v2.pt'))
    p.add_argument('--source-csv', type=Path, default=Path('reports/utility/v2_labels_300_v1/roi_utility.csv'))
    p.add_argument('--image-dir', type=Path, default=Path('data/processed/yolo_flir_lr_x2/images/val'))
    p.add_argument('--frames', type=int, default=24)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.frames < 4 or args.repeats < 2 or args.out.exists():
        p.error('Require --frames >=4, --repeats >=2, and a new --out file')
    if not args.checkpoint.is_file() or not args.source_csv.is_file():
        p.error('Missing checkpoint/source CSV')
    if not args.image_dir.is_dir():
        p.error(f'Image directory not found: {args.image_dir}')
    if not torch.cuda.is_available():
        p.error('CUDA is required to profile the actual GPU deployment path')

    cv2.setNumThreads(1)
    device = torch.device('cuda:0')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if tuple(state.get('heads', ())) != HEADS or list(state.get('input_size', ())) != [80, 64]:
        raise ValueError('Unrecognized predictor architecture')
    if state.get('source_csv_sha256') != file_hash(args.source_csv):
        raise ValueError('Checkpoint/source CSV provenance mismatch')
    model = UtilityPredictor().to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    input_size = tuple(state['input_size'])
    scorer = FastScorer(model, TILES, input_size, device, .25)

    paths = sorted(args.image_dir.glob('*.png'))[:args.frames]
    if len(paths) != args.frames:
        raise ValueError(f'Expected {args.frames} LR images, found {len(paths)}')
    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None or frame.shape != (256, 320):
            raise ValueError(f'Bad LR image: {path}')
        frames.append(frame)

    max_abs_score_diff = 0.0
    for frame in frames:
        x_old, coords_old = reference_encode(frame, TILES, input_size)
        x_fast, coords_fast = fast_encode(frame, TILES, input_size)
        if not torch.equal(x_old, x_fast) or not torch.equal(coords_old, coords_fast):
            raise AssertionError('Encoder changed model inputs! Do not deploy.')
        old = old_scores(model, frame, TILES, input_size, device, .25)
        new = scorer(frame)
        max_abs_score_diff = max(max_abs_score_diff, float(np.max(np.abs(old - new))))
        if not np.allclose(old, new, rtol=0, atol=1e-6):
            raise AssertionError('Prediction scores changed beyond 1e-6! Do not deploy.')
        if top_four(old) != top_four(new):
            raise AssertionError('Selected tiles changed! Do not deploy.')

    print(f'EQUIVALENCE PASSED — {len(frames)} real thermal frames; identical model inputs and top-4 tiles')
    print(f'Max absolute score difference: {max_abs_score_diff:.9g}', flush=True)

    for _ in range(6):
        old_scores(model, frames[0], TILES, input_size, device, .25)
        scorer(frames[0])
    synchronized(device)
    timings = {'original': [], 'optimized': []}
    rng = random.Random(42)
    for _ in range(args.repeats):
        for frame in frames:
            order = ['original', 'optimized']
            rng.shuffle(order)
            for variant in order:
                if variant == 'original':
                    fn = lambda: old_scores(model, frame, TILES, input_size, device, .25)
                else:
                    fn = lambda: scorer(frame)
                timings[variant].append(measure(fn, device))

    old_med = statistics.median(timings['original'])
    fast_med = statistics.median(timings['optimized'])
    delta = old_med - fast_med
    result = {
        'task': 'Step 8B: equivalent utility-selector preprocessing benchmark',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': file_hash(args.checkpoint),
        'source_csv_sha256': file_hash(args.source_csv),
        'frames': len(frames), 'timing_repeats': args.repeats,
        'device': str(device), 'max_abs_score_diff': max_abs_score_diff,
        'identical_inputs_and_top4_on_all_frames': True,
        'original_median_ms': round(old_med, 6),
        'optimized_median_ms': round(fast_med, 6),
        'reduction_ms': round(delta, 6),
        'reduction_percent': round(100 * delta / old_med, 3),
        'note': 'Selector-only synchronized end-to-end inference. No detector or image enhancement. Does not establish full pipeline speedup.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Original selector median:  {old_med:.3f} ms')
    print(f'Optimized selector median: {fast_med:.3f} ms')
    print(f'Selector reduction:        {delta:+.3f} ms ({100*delta/old_med:+.1f}%)')
    print(f'SAVED — {args.out}')


if __name__ == '__main__':
    main()
