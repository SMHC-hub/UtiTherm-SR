"""Step 7A: freeze a sequence-aware FLIR VAL development subset before evaluation.

This validation set was already used in Step 4, so it is *not* an untouched
final test set. It is, however, outside detector/predictor TRAIN image lists
subject to the audited filename/sequence checks below.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sequence_id(name: str) -> str:
    if not name.endswith('.png') or Path(name).name != name:
        raise ValueError(f'Unsafe or unexpected image filename: {name}')
    stem = Path(name).stem
    if '-frame-' not in stem:
        raise ValueError(f'Cannot establish video sequence from filename: {name}')
    return stem.split('-frame-', 1)[0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_sequences(train_dir: Path, utility_csv: Path, split_manifest: Path) -> set[str]:
    if not train_dir.is_dir():
        raise FileNotFoundError(train_dir)
    seqs = {sequence_id(p.name) for p in train_dir.glob('*.png')}
    if not seqs:
        raise ValueError('No detector training images found')
    if not split_manifest.is_file() or not utility_csv.is_file():
        raise FileNotFoundError('Missing predictor split or training ROI measurements')
    split = json.loads(split_manifest.read_text(encoding='utf-8'))
    if split.get('intervention_csv_sha256') != sha256(utility_csv):
        raise ValueError('Utility CSV and predictor split manifest disagree')
    pred_seqs = set(split['train_sequences']) | set(split['val_sequences'])
    if not pred_seqs or not pred_seqs <= seqs:
        raise ValueError('Predictor sequences are inconsistent with detector training set')
    with utility_csv.open(newline='', encoding='utf-8-sig') as fh:
        measured_seqs = {r['sequence'] for r in csv.DictReader(fh)}
    if not measured_seqs or measured_seqs != pred_seqs:
        raise ValueError('Predictor measured sequences do not match train/val split')
    return seqs


def eligible_paths(val_dir: Path, label_dir: Path, exclude_seqs: set[str]) -> tuple[list[Path], dict]:
    if not val_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError('Missing detector validation images or labels')
    images = sorted(val_dir.glob('*.png'))
    if not images:
        raise ValueError('No FLIR validation PNGs found')
    missing_labels = []
    excluded = []
    eligible = []
    for p in images:
        seq = sequence_id(p.name)
        if seq in exclude_seqs:
            excluded.append(p.name)
            continue
        if not (label_dir / (p.stem + '.txt')).is_file():
            missing_labels.append(p.name)
            continue
        eligible.append(p)
    if missing_labels:
        raise ValueError(f'{len(missing_labels)} eligible validation images lack label files; e.g. {missing_labels[0]}')
    return eligible, {'total_val_images': len(images), 'sequence_collisions_excluded': len(excluded),
                      'eligible_images': len(eligible),
                      'eligible_unique_sequences': len({sequence_id(p.name) for p in eligible})}


def select_diverse(paths: list[Path], n: int, seed: int, max_per_sequence: int) -> list[Path]:
    if n < 1 or max_per_sequence < 1:
        raise ValueError('Need positive number of images and max-per-sequence')
    grouped: dict[str,list[Path]] = defaultdict(list)
    for p in paths:
        grouped[sequence_id(p.name)].append(p)
    rng = random.Random(seed)
    keys = sorted(grouped)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(grouped[key])
    selected = []
    for turn in range(max_per_sequence):
        for key in keys:
            if turn < len(grouped[key]):
                selected.append(grouped[key][turn])
                if len(selected) == n:
                    return selected
    return selected


def check_manifest(m: dict, val_dir: Path, label_dir: Path,
                   training_seq: set[str], utility_csv: Path,
                   split_manifest: Path) -> list[str]:
    if m['protocol'] != 'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST':
        raise ValueError('Unsafe or unknown split protocol')
    if m['utility_csv_sha256'] != sha256(utility_csv) or m['predictor_split_sha256'] != sha256(split_manifest):
        raise ValueError('The training protocol sources changed after split freezing')
    if Path(m['eval_lr_directory']).resolve() != val_dir.resolve():
        raise ValueError('The evaluation directory changed after split freezing')
    names = m['images']
    if len(names) != m['selected_images'] or not names or len(names) != len(set(names)):
        raise ValueError('Invalid or duplicate selected image names')
    counts = defaultdict(int)
    for n in names:
        seq = sequence_id(n)
        if seq in training_seq:
            raise ValueError(f'Detector TRAIN and development VAL share a sequence: {seq}')
        counts[seq] += 1
        if counts[seq] > m['max_per_sequence']:
            raise ValueError('Selected too many frames from one sequence')
        image_path, label_path = val_dir/n, label_dir/(Path(n).stem+'.txt')
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f'Split image or label no longer exists: {n}')
        if m['image_sha256'][n] != sha256(image_path) or m['label_sha256'][n] != sha256(label_path):
            raise ValueError(f'Frozen image or annotation contents changed: {n}')
    if len(counts) != m['unique_sequences']:
        raise ValueError('Sequence counts changed')
    return names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/development_eval.yaml'))
    p.add_argument('--target', type=int, default=200)
    p.add_argument('--out', type=Path, default=Path('reports/development/split_200_v1'))
    a = p.parse_args()
    if a.out.exists():
        p.error('Output directory already exists; use a new --out')
    cfg = yaml.safe_load(a.config.read_text(encoding='utf-8'))
    val_dir = Path(cfg['eval_lr']); train_dir = Path(cfg['train_lr'])
    label_dir = val_dir.parent.parent/'labels'/'val'
    csv_path, split_path = Path(cfg['utility_csv']), Path(cfg['split_manifest'])
    train_seqs = training_sequences(train_dir, csv_path, split_path)
    eligible, stats = eligible_paths(val_dir, label_dir, train_seqs)
    max_per = int(cfg.get('max_per_sequence', 3))
    sample = select_diverse(eligible, a.target, int(cfg['seed']), max_per)
    print(f'VAL audit: {stats}; selected {len(sample)}/{a.target} across '
          f'{len({sequence_id(p.name) for p in sample})} sequences')
    if len(sample) < a.target:
        raise SystemExit(f'ERROR: Cannot select {a.target} images with <= {max_per} frames/sequence. '
                         'Inspect the audit, then adjust target/max_per_sequence deliberately.')
    if len(sample) < 100:
        raise SystemExit('At least 100 images required for this extended evaluation')
    ckpt_keys = ('predictor_checkpoint', 'detector_checkpoint', 'sr_checkpoint')
    for key in ckpt_keys:
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(f'Missing frozen model checkpoint: {cfg[key]}')
    manifest = {
        'protocol': 'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': int(cfg['seed']), 'max_per_sequence': max_per,
        'selected_images': len(sample),
        'unique_sequences': len({sequence_id(p.name) for p in sample}),
        'dataset_counts': stats, 'images': [p.name for p in sample],
        'eval_lr_directory': str(val_dir.resolve()),
        'image_sha256': {p.name:sha256(p) for p in sample},
        'label_sha256': {p.name:sha256(label_dir/(p.stem+'.txt')) for p in sample},
        'utility_csv_sha256': sha256(csv_path),
        'predictor_split_sha256': sha256(split_path),
        'checkpoint_sha256': {k: sha256(Path(cfg[k])) for k in ckpt_keys},
        'notes': [
            'No image or video sequence overlaps the enumerated detector TRAIN set.',
            'Selection is label-blind, frozen before evaluating predictor policies.',
            'FLIR VAL has previously guided development (Step 4): NOT untouched final-test data.',
            'Sequence IDs are parsed from filenames; physical-scene and near-duplicate checks remain necessary.',
            'Ground truth is only read after model inference for diagnostic metrics.',
        ],
    }
    check_manifest(manifest, val_dir, label_dir, train_seqs, csv_path, split_path)
    a.out.mkdir(parents=True)
    f = a.out/'manifest.json'
    f.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'FROZEN — {f} | {len(sample)} images, {manifest["unique_sequences"]} sequences')


if __name__ == '__main__':
    main()
