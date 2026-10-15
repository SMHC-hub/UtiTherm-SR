"""Step 9D3: divergence-guarded Large FSRCNN x2 training on official FLIR TRAIN.

Run from UtiTherm-SR root (Docker research service):
  python scripts/train_fsrcnn_large_stable_step9d3.py --smoke
  python scripts/train_fsrcnn_large_stable_step9d3.py

Reuses the Step 4B ThermalPairs, group_split, evaluation, and checkpoint
format. Preserves the original and diverged Step 9D1 checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from train_fsrcnn import (SUFFIXES, ThermalPairs, atomic_checkpoint, evaluate,
                          group_split, mse_psnr, sequence_id, sha256_names)
from utitherm.sr.fsrcnn import FSRCNN

CONFIG_DEFAULT = Path('configs/fsrcnn_large_x2_stable_step9d3.yaml')
REQUIRED = {'train_hr', 'output', 'canonical_checkpoint', 'epochs', 'batch', 'lr_patch',
            'learning_rate', 'seed', 'val_fraction', 'workers', 'patience', 'd', 's', 'm',
            'max_grad_norm', 'max_batch_mse', 'lr_plateau_patience', 'lr_plateau_factor',
            'min_learning_rate'}
ARCH = {'d': 168, 's': 36, 'm': 10}
PROTECTED_CHECKPOINTS = {
    Path('models/sr/fsrcnn_x2_v1.pt'),
    Path('models/sr/fsrcnn_large_x2_step9d_v1.pt'),
}


def validate_config(cfg: dict) -> dict:
    """Validate all known training guardrails without touching disk."""
    if not isinstance(cfg, dict):
        raise ValueError('Config must be a YAML mapping')
    missing = REQUIRED - cfg.keys()
    if missing:
        raise ValueError(f'Missing configuration keys: {sorted(missing)}')
    if any(int(cfg[k]) != ARCH[k] for k in ARCH):
        raise ValueError('Step 9D3 must keep the exact Large FSRCNN d/s/m architecture')
    if int(cfg['epochs']) < 1 or int(cfg['batch']) < 1 or int(cfg['lr_patch']) < 8:
        raise ValueError('Invalid epochs, batch, or patch size')
    if int(cfg['workers']) < 0 or int(cfg['patience']) < 1 or int(cfg['lr_plateau_patience']) < 1:
        raise ValueError('Invalid workers or patience')
    if not (0 < float(cfg['val_fraction']) < .5):
        raise ValueError('Invalid validation fraction')
    lr = float(cfg['learning_rate'])
    if not (0 < lr <= .0001):
        raise ValueError('Step 9D3 learning rate must be positive and <=1e-4')
    if not (0 < float(cfg['min_learning_rate']) <= lr):
        raise ValueError('Invalid min learning rate')
    if not (0 < float(cfg['lr_plateau_factor']) < 1):
        raise ValueError('Plateau LR factor must be between zero and one')
    if not (0 < float(cfg['max_grad_norm']) <= 5):
        raise ValueError('Invalid gradient clipping norm')
    if not (0 < float(cfg['max_batch_mse']) <= 10):
        raise ValueError('Invalid training loss guard')
    source = Path(str(cfg['train_hr']))
    if source.name != 'train' or source.parent.name != 'images':
        raise ValueError('FLIR source must be images/train, never official validation')
    if Path(cfg['output']) == Path('reports/sr/fsrcnn_large_x2_step9d_v1'):
        raise ValueError('Refusing to overwrite the unstable Step 9D1 experiment')
    canonical = Path(str(cfg['canonical_checkpoint']))
    if canonical in PROTECTED_CHECKPOINTS or canonical.name != 'fsrcnn_large_x2_stable_step9d3_v1.pt':
        raise ValueError('Specify the separate Step 9D3 canonical checkpoint')
    if Path(cfg['output']).name != 'fsrcnn_large_x2_stable_step9d3_v1':
        raise ValueError('Specify the separate Step 9D3 report directory')
    return cfg


def enforce_stable_loss(loss: torch.Tensor, ceiling: float, epoch: int, step: int) -> float:
    """Reject runaway normalized-range batch MSE before backward/optimizer step."""
    value = float(loss.detach().item())
    if not math.isfinite(value) or value > ceiling or value < 0:
        raise RuntimeError(
            f'UNSTABLE LOSS epoch={epoch} batch={step} mse={value} ceiling={ceiling}; '
            'no optimizer step performed for this batch. Investigate optimizer/model initialization.'
        )
    return value


def checked_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                 lr: torch.Tensor, hr: torch.Tensor, *,
                 max_batch_mse: float, max_grad_norm: float,
                 epoch: int, step: int) -> tuple[float, float]:
    """One guarded optimizer update; never silently skip an exploding batch."""
    prediction = model(lr)
    if tuple(prediction.shape) != tuple(hr.shape):
        raise ValueError('Unexpected output/target shapes')
    loss = torch.mean((prediction - hr).square())
    value = enforce_stable_loss(loss, max_batch_mse, epoch, step)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm,
                                    error_if_nonfinite=True)
    optimizer.step()
    return value, float(norm.detach().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_DEFAULT)
    parser.add_argument('--smoke', action='store_true', help='One epoch, 4 train batches, 2 validation batches; separate output')
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f'Missing configuration: {args.config}')
    try:
        cfg = validate_config(yaml.safe_load(args.config.read_text(encoding='utf-8')))
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    source = Path(str(cfg['train_hr']))
    if not source.is_dir():
        parser.error(f'Missing FLIR train folder: {source}')
    paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    if len(paths) < 2:
        parser.error('Not enough training images')
    seed = int(cfg['seed'])
    train_files, val_files = group_split(paths, seed, float(cfg['val_fraction']))
    actual_train = train_files[:64] if args.smoke else train_files
    actual_val = val_files[:16] if args.smoke else val_files
    run_root = Path(str(cfg['output']))
    out = run_root.with_name(run_root.name + '_smoke') if args.smoke else run_root
    canonical = Path(str(cfg['canonical_checkpoint']))
    if out.exists():
        parser.error(f'Report directory already exists; refusing overwrite: {out}')
    if not args.smoke and canonical.exists():
        parser.error(f'Canonical checkpoint already exists; refusing overwrite: {canonical}')
    if not torch.cuda.is_available():
        parser.error('Step 9D3 training requires CUDA')

    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device('cuda:0')
    ds_train = ThermalPairs(actual_train, int(cfg['lr_patch']), True, seed)
    ds_val = ThermalPairs(actual_val, int(cfg['lr_patch']), False, seed)
    workers = int(cfg['workers'])
    bs = int(cfg['batch'])
    tr = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=workers,
                    pin_memory=True, persistent_workers=bool(workers))
    va = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=workers,
                    pin_memory=True, persistent_workers=bool(workers))
    model_args = {k: int(cfg[k]) for k in ('d', 's', 'm')}
    model = FSRCNN(scale=2, **model_args).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    if num_params != 148009:
        raise RuntimeError(f'Unexpected FSRCNN parameter count: {num_params}')
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['learning_rate']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=float(cfg['lr_plateau_factor']),
        patience=int(cfg['lr_plateau_patience']),
        min_lr=float(cfg['min_learning_rate'])
    )
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    manifest = {
        'task': 'Step 9D3 stable Large FSRCNN x2 thermal SR training',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(source), 'smoke': args.smoke, 'architecture': {'scale': 2, **model_args},
        'parameters': num_params, 'config': cfg, 'config_sha256': config_sha,
        'training_images': len(actual_train), 'internal_validation_images': len(actual_val),
        'train_sequence_count': len({sequence_id(p) for p in actual_train}),
        'val_sequence_count': len({sequence_id(p) for p in actual_val}),
        'train_names_sha256': sha256_names(actual_train),
        'val_names_sha256': sha256_names(actual_val),
        'full_train_images': len(train_files), 'full_val_images': len(val_files),
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
        'note': 'train-domain internal validation patches; NOT official FLIR validation nor independent test',
        'numerical_policy': 'gradient clipping + fail-fast on batch loss, finite grads + LR plateau',
    }
    out.mkdir(parents=True)
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'GPU: {manifest["gpu"]} | params={num_params:,} | stable LR={cfg["learning_rate"]} | clip={cfg["max_grad_norm"]}', flush=True)
    print(f'Train/held-out: {len(actual_train)}/{len(actual_val)} | output={out}', flush=True)
    epochs = 1 if args.smoke else int(cfg['epochs'])
    best_mse, stale = float('inf'), 0
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        sum_loss, sum_pixels, largest_grad = 0.0, 0, 0.0
        for step, (lr, hr) in enumerate(tr, 1):
            if args.smoke and step > 4:
                break
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            lval, grad_norm = checked_step(
                model, optimizer, lr, hr,
                max_batch_mse=float(cfg['max_batch_mse']),
                max_grad_norm=float(cfg['max_grad_norm']), epoch=epoch, step=step,
            )
            largest_grad = max(largest_grad, grad_norm)
            sum_loss += lval * hr.numel()
            sum_pixels += hr.numel()
        train_mse = sum_loss / max(1, sum_pixels)
        val_mse = evaluate(model, va, device, max_batches=2 if args.smoke else 0)
        if not math.isfinite(val_mse) or val_mse < 0 or val_mse > float(cfg['max_batch_mse']):
            raise RuntimeError(f'UNSTABLE VALIDATION epoch={epoch}, MSE={val_mse}; abort before checkpoint')
        scheduler.step(val_mse)
        improvement = val_mse < best_mse - 1e-9
        if improvement:
            best_mse, stale = val_mse, 0
        else:
            stale += 1
        model_checkpoint = {
            'state_dict': model.state_dict(),
            'model_kwargs': {'scale': 2, **model_args},
            'epoch': epoch, 'val_mse': val_mse, 'val_psnr_db': mse_psnr(val_mse),
            'train_split_sha256': sha256_names(actual_train),
            'config_sha256': config_sha,
        }
        atomic_checkpoint(out / 'last.pt', model_checkpoint)
        if improvement:
            atomic_checkpoint(out / 'best.pt', model_checkpoint)
        metrics = {
            'epoch': epoch, 'train_mse': round(train_mse, 8),
            'val_mse': round(val_mse, 8), 'val_psnr_db': round(mse_psnr(val_mse), 4),
            'lr': optimizer.param_groups[0]['lr'],
            'max_preclip_grad_norm': round(largest_grad, 5),
            'epoch_seconds': round(time.perf_counter() - start, 2),
        }
        logs.append(metrics)
        with (out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + '\n')
        print(f'epoch {epoch:02}/{epochs}: train MSE={train_mse:.6f} held-out PSNR={mse_psnr(val_mse):.2f}dB '
              f'best={mse_psnr(best_mse):.2f}dB LR={optimizer.param_groups[0]["lr"]:.1e} '
              f'preclip_grad_max={largest_grad:.2f}', flush=True)
        if stale >= int(cfg['patience']):
            print(f'Early stopping after {stale} epochs without validation improvement', flush=True)
            break
    manifest['best_internal_val_psnr_db'] = round(mse_psnr(best_mse), 4)
    manifest['epochs_completed'] = len(logs)
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    if not args.smoke:
        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / 'best.pt', canonical)
        print(f'Canonical checkpoint: {canonical}', flush=True)
    print(f'DONE — best={mse_psnr(best_mse):.2f}dB | {out / "best.pt"}', flush=True)


if __name__ == '__main__':
    main()
