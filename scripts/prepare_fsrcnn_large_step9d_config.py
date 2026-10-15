"""Generate a Large-FSRCNN config from the exact working FSRCNN baseline YAML.

Do not guess FLIR paths or baseline training hyperparameters. The generated YAML
is a new file; the source config and all checkpoints remain unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import yaml

D, S, M = 168, 36, 10
OUTPUT = 'reports/sr/fsrcnn_large_x2_step9d_v1'
CHECKPOINT = 'models/sr/fsrcnn_large_x2_step9d_v1.pt'
CONFIG = 'configs/fsrcnn_large_x2_step9d.yaml'
REQUIRED = {
    'train_hr', 'output', 'canonical_checkpoint', 'epochs', 'batch', 'lr_patch',
    'learning_rate', 'seed', 'val_fraction', 'workers', 'patience', 'd', 's', 'm'
}


def make_large_config(baseline: dict) -> dict:
    if not isinstance(baseline, dict):
        raise ValueError('Baseline YAML must be a mapping')
    missing = REQUIRED - set(baseline)
    if missing:
        raise ValueError(f'Baseline config missing: {sorted(missing)}')
    if Path(str(baseline['train_hr'])).name != 'train' or Path(str(baseline['train_hr'])).parent.name != 'images':
        raise ValueError('Expected baseline train_hr to be an images/train directory')
    if int(baseline['batch']) < 1:
        raise ValueError('Baseline batch must be positive')
    if Path(str(baseline['canonical_checkpoint'])) == Path(CHECKPOINT):
        raise ValueError('Baseline checkpoint unexpectedly points to the Large run')
    if Path(str(baseline['output'])) == Path(OUTPUT):
        raise ValueError('Baseline output unexpectedly points to the Large run')
    conf = deepcopy(baseline)
    conf.update({'d': D, 's': S, 'm': M, 'output': OUTPUT,
                 'canonical_checkpoint': CHECKPOINT,
                 # Conservative default for RTX 5070 12-GB VRAM; can be tuned after a smoke run.
                 'batch': min(int(baseline['batch']), 8)})
    return conf


def prepare(baseline_path: Path, target_path: Path) -> dict:
    if not baseline_path.is_file():
        raise FileNotFoundError(f'Missing working baseline config: {baseline_path}')
    if target_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing config: {target_path}')
    baseline = yaml.safe_load(baseline_path.read_text(encoding='utf-8'))
    conf = make_large_config(baseline)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation, even if file appears between checks.
    with target_path.open('x', encoding='utf-8') as f:
        f.write('# Generated from configs/fsrcnn_x2.yaml; do not overwrite the original.\n')
        yaml.safe_dump(conf, f, sort_keys=False, allow_unicode=True)
    return conf


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('configs/fsrcnn_x2.yaml'))
    p.add_argument('--out', type=Path, default=Path(CONFIG))
    args = p.parse_args()
    try:
        conf = prepare(args.base, args.out)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        p.error(str(exc))
    print(f'CREATED: {args.out}')
    print(f'Large FSRCNN x2: d={conf["d"]}, s={conf["s"]}, m={conf["m"]} | batch={conf["batch"]}')
    print(f'FLIR TRAIN images: {conf["train_hr"]}')
    print(f'Run output: {conf["output"]}')
    print(f'Canonical checkpoint: {conf["canonical_checkpoint"]}')
    print('Original training YAML, model and evaluation artifacts unchanged.')


if __name__ == '__main__':
    main()
