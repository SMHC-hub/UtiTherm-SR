"""Freeze a fresh sequence-aware sample for Step 7B2A utility-label collection.

All files come from detector TRAIN images. Excludes ALL Step 5B V1
sequence IDs, preventing V2 label rows from silently duplicating V1 samples.
A separate final untouched test set is still required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROTOCOL = 'STEP_7B2A_FRESH_TRAIN_SEQUENCES_NOT_FINAL_TEST'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sequence_id(filename: str) -> str:
    p = Path(filename)
    if p.name != filename or '\\' in filename or p.suffix.lower() != '.png' or '-frame-' not in p.stem:
        raise ValueError(f'Unexpected or unsafe FLIR image filename: {filename!r}')
    return p.stem.split('-frame-', 1)[0]


def v1_sequences(csv_path: Path, split_path: Path) -> set[str]:
    if not csv_path.is_file() or not split_path.is_file():
        raise FileNotFoundError('Cannot audit V1 sequences: missing ROI CSV or V1 split manifest')
    split = json.loads(split_path.read_text(encoding='utf-8'))
    if split.get('intervention_csv_sha256') != sha256(csv_path):
        raise ValueError('V1 split manifest does not match V1 ROI CSV hash')
    recorded = set(split['train_sequences']) | set(split['val_sequences'])
    if not recorded or set(split['train_sequences']) & set(split['val_sequences']):
        raise ValueError('V1 split contains missing or overlapping sequences')
    with csv_path.open(newline='', encoding='utf-8-sig') as handle:
        measured = {r['sequence'] for r in csv.DictReader(handle)}
    if measured != recorded:
        raise ValueError('V1 ROI CSV sequence IDs disagree with the V1 split')
    return recorded


def audited_candidates(train_dir: Path, val_dir: Path, excludes: set[str]):
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError('Expected detector train and validation image directories')
    train_labels = train_dir.parent.parent / 'labels' / 'train'
    if not train_labels.is_dir():
        raise FileNotFoundError(train_labels)
    val_seqs = {sequence_id(p.name) for p in val_dir.glob('*.png')}
    paths = sorted(train_dir.glob('*.png'))
    if not paths:
        raise ValueError('No detector training PNG images')
    rejected_v1 = rejected_val = 0
    candidates = []
    for path in paths:
        seq = sequence_id(path.name)
        if seq in excludes:
            rejected_v1 += 1
            continue
        if seq in val_seqs:
            rejected_val += 1
            continue
        if not (train_labels / f'{path.stem}.txt').is_file():
            raise FileNotFoundError(f'Missing TRAIN label for candidate: {path.name}')
        candidates.append(path)
    audit = dict(total_train_images=len(paths), v1_sequence_frames_excluded=rejected_v1,
                 validation_sequence_frames_excluded=rejected_val,
                 eligible_images=len(candidates), eligible_sequences=len({sequence_id(p.name) for p in candidates}),
                 validation_sequences=len(val_seqs))
    return candidates, audit


def select_round_robin(paths: list[Path], target: int, seed: int, max_per_seq: int) -> list[Path]:
    if target < 1 or max_per_seq < 1:
        raise ValueError('target and max_per_sequence must be positive')
    grouped = defaultdict(list)
    for p in paths:
        grouped[sequence_id(p.name)].append(p)
    rng = random.Random(seed)
    keys = sorted(grouped)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(grouped[key])
    selected = []
    for j in range(max_per_seq):
        for key in keys:
            if len(grouped[key]) > j:
                selected.append(grouped[key][j])
                if len(selected) == target:
                    return selected
    return selected


def verify_manifest(manifest: dict, cfg: dict, verify_hashes: bool = True) -> list[Path]:
    if manifest.get('protocol') != PROTOCOL:
        raise ValueError('Unknown utility-data protocol')
    train_dir = Path(cfg['train_lr'])
    val_dir = Path(cfg['val_lr'])
    excludes = v1_sequences(Path(cfg['v1_roi_csv']), Path(cfg['v1_split_manifest']))
    if manifest['v1_csv_sha256'] != sha256(Path(cfg['v1_roi_csv'])):
        raise ValueError('V1 ROI source has changed')
    if manifest['v1_split_sha256'] != sha256(Path(cfg['v1_split_manifest'])):
        raise ValueError('V1 split has changed')
    if manifest['detector_checkpoint_sha256'] != sha256(Path(cfg['detector_checkpoint'])):
        raise ValueError('Detector checkpoint has changed')
    if Path(manifest['train_lr']).resolve() != train_dir.resolve():
        raise ValueError('TRAIN root differs from frozen manifest')
    val_seqs = {sequence_id(p.name) for p in val_dir.glob('*.png')}
    names = manifest['images']
    if (not names or len(names) != manifest['num_images'] or
            len(names) != len(set(names))):
        raise ValueError('Manifest image list empty, duplicate, or count mismatch')
    counts = Counter()
    all_paths = []
    label_dir = train_dir.parent.parent / 'labels' / 'train'
    for name in names:
        seq = sequence_id(name)
        if seq in excludes or seq in val_seqs:
            raise ValueError(f'Invalid V1/VAL sequence overlap: {seq}')
        counts[seq] += 1
        if counts[seq] > manifest['max_per_sequence']:
            raise ValueError(f'Exceeded per-sequence cap: {seq}')
        img = train_dir / name
        label = label_dir / f'{img.stem}.txt'
        if not img.is_file() or not label.is_file():
            raise FileNotFoundError(f'Missing frozen image/label: {name}')
        if verify_hashes and (manifest['image_sha256'][name] != sha256(img) or
                              manifest['label_sha256'][name] != sha256(label)):
            raise ValueError(f'Frozen source contents changed: {name}')
        all_paths.append(img)
    if len(counts) != manifest['unique_sequences']:
        raise ValueError('Manifest unique-sequence count mismatch')
    return all_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=Path('configs/utility_v2_data.yaml'))
    ap.add_argument('--target', type=int, default=300)
    ap.add_argument('--out', type=Path, default=Path('reports/utility/v2_fresh_split_300_v1'))
    args = ap.parse_args()
    if args.out.exists():
        ap.error('Output already exists. Use a fresh --out to preserve reproducibility.')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    ckpt = Path(cfg['detector_checkpoint'])
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    exclusions = v1_sequences(Path(cfg['v1_roi_csv']), Path(cfg['v1_split_manifest']))
    paths, audit = audited_candidates(Path(cfg['train_lr']), Path(cfg['val_lr']), exclusions)
    chosen = select_round_robin(paths, args.target, int(cfg['seed']), int(cfg['max_per_sequence']))
    unique_count = len({sequence_id(p.name) for p in chosen})
    print(f'TRAIN audit: {audit}; selected {len(chosen)}/{args.target} across {unique_count} NEW sequences', flush=True)
    if len(chosen) != args.target:
        raise SystemExit('ERROR: Too few eligible files under current sequence cap. Do not silently relax exclusion rules.')
    if args.target >= 100 and unique_count < int(cfg['min_unique_sequences']):
        raise SystemExit('ERROR: Selected too few unique sequences for larger collection; audit before proceeding.')
    label_dir = Path(cfg['train_lr']).parent.parent / 'labels' / 'train'
    manifest = dict(
        protocol=PROTOCOL, created_utc=datetime.now(timezone.utc).isoformat(),
        train_lr=str(Path(cfg['train_lr']).resolve()), seed=int(cfg['seed']),
        max_per_sequence=int(cfg['max_per_sequence']), num_images=len(chosen),
        unique_sequences=unique_count, images=[p.name for p in chosen],
        image_sha256={p.name:sha256(p) for p in chosen},
        label_sha256={p.name:sha256(label_dir/f'{p.stem}.txt') for p in chosen},
        v1_csv_sha256=sha256(Path(cfg['v1_roi_csv'])),
        v1_split_sha256=sha256(Path(cfg['v1_split_manifest'])),
        detector_checkpoint_sha256=sha256(ckpt), audit=audit,
        notes=['Only fresh detector-TRAIN sequences, excluding every V1 utility-probe sequence.',
               'No detector VAL frames used as utility training targets.',
               'The detector was trained on these TRAIN frames: target optimism remains.',
               'Physical scene independence not proven solely by filename sequence IDs.',
               'This is NOT independent test data and does not address out-of-fold labeling.']
    )
    verify_manifest(manifest, cfg)
    args.out.mkdir(parents=True)
    result = args.out/'manifest.json'
    result.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'FROZEN — {result} | {len(chosen)} images, {unique_count} sequences', flush=True)


if __name__ == '__main__':
    main()
