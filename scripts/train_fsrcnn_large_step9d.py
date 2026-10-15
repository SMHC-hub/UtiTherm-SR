"""Step 9D: run existing, audited FSRCNN trainer with a NEW Large-model config.

Keeps the original train_fsrcnn.py untouched and uses its grouped split,
synthetic degradation, early stopping, logging and checkpoint format.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/fsrcnn_large_x2_step9d.yaml'
BASE_CONFIG = ROOT / 'configs/fsrcnn_x2.yaml'


def validate_large_config(config: dict, baseline: dict) -> None:
    if not isinstance(config, dict) or not isinstance(baseline, dict):
        raise ValueError('Configuration must be a YAML mapping')
    if (config.get('d'), config.get('s'), config.get('m')) != (168, 36, 10):
        raise ValueError('Step 9D must use d=168, s=36, m=10 (148,009 parameters)')
    for key in ('train_hr', 'seed', 'val_fraction', 'lr_patch', 'learning_rate',
                'epochs', 'patience', 'workers'):
        if config.get(key) != baseline.get(key):
            raise ValueError(f'Unexpected change from original FSRCNN training protocol: {key}')
    if not (1 <= config.get('batch', 0) <= 8):
        raise ValueError('Step 9D batch must be 1..8 for the initial VRAM-safe run')
    for key in ('output', 'canonical_checkpoint'):
        if config.get(key) == baseline.get(key):
            raise ValueError(f'Step 9D must not reuse baseline {key}')
    train_hr = Path(str(config.get('train_hr', '')))
    if train_hr.name != 'train' or train_hr.parent.name != 'images':
        raise ValueError('Step 9D must use official FLIR images/train, never FLIR validation')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--baseline-config', type=Path, default=BASE_CONFIG)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    for path in (args.config, args.baseline_config):
        if not path.is_file():
            p.error(f'Missing configuration: {path}. Generate the Large config first.')
    try:
        config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        base = yaml.safe_load(args.baseline_config.read_text(encoding='utf-8'))
        validate_large_config(config, base)
    except (ValueError, yaml.YAMLError) as exc:
        p.error(str(exc))
    # Forward to the existing, unmodified trainer. That trainer itself refuses to
    # overwrite any existing report directory or canonical model checkpoint.
    import train_fsrcnn
    original_argv = sys.argv
    sys.argv = [str(ROOT / 'scripts/train_fsrcnn.py'), '--config', str(args.config)]
    if args.smoke:
        sys.argv.append('--smoke')
    try:
        train_fsrcnn.main()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
