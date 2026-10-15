"""CPU-only Step 9D3 safeguards. Invoke from project root: python -m pytest -q tests/test_fsrcnn_large_stable_step9d3.py"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from train_fsrcnn_large_stable_step9d3 import (validate_config, enforce_stable_loss,
                                                 checked_step, CONFIG_DEFAULT, ARCH)
from train_fsrcnn import ThermalPairs, group_split, sequence_id
from utitherm.sr.fsrcnn import FSRCNN


@pytest.fixture
def cfg():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / CONFIG_DEFAULT).read_text(encoding='utf-8'))


def test_config_valid(cfg):
    assert validate_config(cfg) is cfg
    assert cfg['learning_rate'] == 1e-4


def test_keep_exact_architecture(cfg):
    for key in ARCH:
        bad = copy.deepcopy(cfg)
        bad[key] += 1
        with pytest.raises(ValueError, match='architecture'):
            validate_config(bad)


def test_reject_large_lr(cfg):
    cfg['learning_rate'] = .001
    with pytest.raises(ValueError, match='learning rate'):
        validate_config(cfg)


def test_reject_zero_lr(cfg):
    cfg['learning_rate'] = 0
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_reject_unstable_loss_limit(cfg):
    cfg['max_batch_mse'] = 11
    with pytest.raises(ValueError, match='loss guard'):
        validate_config(cfg)


def test_reject_clip_norm(cfg):
    cfg['max_grad_norm'] = -1
    with pytest.raises(ValueError, match='gradient'):
        validate_config(cfg)


def test_reject_val_source(cfg):
    cfg['train_hr'] = 'data/processed/yolo_flir/images/val'
    with pytest.raises(ValueError, match='images/train'):
        validate_config(cfg)


def test_reject_failed_checkpoint_overwrite(cfg):
    cfg['canonical_checkpoint'] = 'models/sr/fsrcnn_large_x2_step9d_v1.pt'
    with pytest.raises(ValueError, match='Step 9D3'):
        validate_config(cfg)


def test_reject_failed_report_overwrite(cfg):
    cfg['output'] = 'reports/sr/fsrcnn_large_x2_step9d_v1'
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_reject_missing_config_key(cfg):
    cfg.pop('max_grad_norm')
    with pytest.raises(ValueError, match='Missing'):
        validate_config(cfg)


def test_reject_non_mapping():
    with pytest.raises(ValueError):
        validate_config(['not a config'])


@pytest.mark.parametrize('bad', [float('inf'), float('-inf'), float('nan'), 10.00001, -0.1])
def test_spike_guard_aborts(bad):
    with pytest.raises(RuntimeError, match='UNSTABLE LOSS'):
        enforce_stable_loss(torch.tensor(bad), 10., 4, 7)


def test_spike_guard_allows_normal_batch():
    assert abs(enforce_stable_loss(torch.tensor(.125), 10., 1, 1) - .125) < 1e-5


def test_checked_step_updates_with_clipping():
    torch.manual_seed(1)
    model = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x, y = torch.ones(2, 1, 4, 4), torch.zeros(2, 1, 4, 4)
    before = model.weight.detach().clone()
    loss, norm = checked_step(model, optimizer, x, y, max_batch_mse=10,
                              max_grad_norm=.1, epoch=1, step=1)
    assert math.isfinite(loss) and math.isfinite(norm)
    assert not torch.equal(before, model.weight.detach())
    # The gradients remaining after clip must have norm <= prescribed cap.
    post_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000).item()
    assert post_norm <= .10001


def test_checked_step_rejects_spike_before_update():
    model = torch.nn.Conv2d(1, 1, 1)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.ones(1, 1, 4, 4)
    target = torch.full_like(x, 100.)
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(RuntimeError, match='UNSTABLE LOSS'):
        checked_step(model, optim, x, target, max_batch_mse=10,
                     max_grad_norm=1., epoch=1, step=2)
    assert all(torch.equal(old, cur.detach()) for old, cur in zip(before, model.parameters()))


def test_fsrsnn_checkpoint_compatible_shape_and_param_count():
    model = FSRCNN(scale=2, d=168, s=36, m=10)
    assert sum(p.numel() for p in model.parameters()) == 148009
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 8, 8))
    assert output.shape == (1, 1, 16, 16)


def test_group_split_no_sequence_overlap(tmp_path):
    paths = [tmp_path / f'video-{s}-frame-{i:06d}.jpg'
             for s in ('alpha','beta','gamma','delta') for i in range(3)]
    tr, va = group_split(paths, 42, .25)
    assert tr and va
    assert {sequence_id(p) for p in tr}.isdisjoint({sequence_id(p) for p in va})


def test_thermal_pairs_shape_and_range(tmp_path):
    f = tmp_path / 'video-one-frame-000010.png'
    Image.fromarray(np.arange(128*128, dtype=np.uint32).reshape(128,128).astype('uint8')).save(f)
    ds = ThermalPairs([f], lr_patch=16, training=False, seed=42)
    lr, hr = ds[0]
    assert lr.shape == (1,16,16) and hr.shape == (1,32,32)
    assert float(lr.min()) >= 0 and float(hr.max()) <= 1
    lr2, hr2 = ds[0]
    assert torch.equal(lr, lr2) and torch.equal(hr,hr2)
