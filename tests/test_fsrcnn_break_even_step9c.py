"""Step 9C fast CPU-only unit tests; do not require a CUDA device or FLIR data."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'benchmark_fsrcnn_break_even_step9c.py'
spec = importlib.util.spec_from_file_location('step9c_benchmark', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config(tmp_path, **overrides):
    cfg = {
        'data_config': 'configs/selective_sr_eval.yaml',
        'sr_checkpoint': 'models/sr/fsrcnn_x2_v1.pt',
        'frames': 8, 'repeats': 5, 'warmup': 3, 'seed': 42,
        'selector_ms': 1.107,
        'candidates': [{'name': 'medium_untrained', 'scale': 2, 'd': 112, 's': 24, 'm': 8}],
    }
    cfg.update(overrides)
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    return path


def test_real_config_loads():
    loaded = module.load_settings(ROOT / 'configs' / 'fsrcnn_break_even_step9c.yaml')
    assert loaded['frames'] == 8
    assert loaded['selector_ms'] == 1.107
    assert len(loaded['candidates']) == 3
    assert loaded['candidates'][2][1]['d'] == 224


def test_default_architecture_grid_has_sixteen_tiles():
    assert len(module.TILES) == 16
    assert module.FIXED_FOUR == [0, 5, 10, 15]
    assert module.TILES[0] == (0, 0, 160, 128)
    assert module.TILES[-1] == (480, 384, 640, 512)


def test_missing_config_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.load_settings(tmp_path / 'missing.yaml')


def test_missing_settings_rejected(tmp_path):
    path = config(tmp_path)
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    del data['repeats']
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    with pytest.raises(ValueError, match='required'):
        module.load_settings(path)


@pytest.mark.parametrize('bad', [0, -1, 'five', 2.5, True])
def test_invalid_repeats_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match='repeats'):
        module.load_settings(config(tmp_path, repeats=bad))


@pytest.mark.parametrize('bad', [-0.1, 1000, '1.1', float('nan')])
def test_invalid_selector_time_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match='selector_ms'):
        module.load_settings(config(tmp_path, selector_ms=bad))


def test_duplicate_candidates_rejected(tmp_path):
    a = {'name': 'medium_untrained', 'scale': 2, 'd': 112, 's': 24, 'm': 8}
    with pytest.raises(ValueError, match='unique'):
        module.load_settings(config(tmp_path, candidates=[a, a.copy()]))


def test_untrained_tag_required(tmp_path):
    a = {'name': 'trained_medium', 'scale': 2, 'd': 112, 's': 24, 'm': 8}
    with pytest.raises(ValueError, match='untrained'):
        module.load_settings(config(tmp_path, candidates=[a]))


def test_invalid_scale_rejected(tmp_path):
    a = {'name': 'bad_untrained', 'scale': 4, 'd': 112, 's': 24, 'm': 8}
    with pytest.raises(ValueError, match='architecture'):
        module.load_settings(config(tmp_path, candidates=[a]))


def test_break_even_true():
    calc = module.break_even_estimate(5.0, 2.0, 1.1)
    assert calc['potential_break_even'] is True
    assert calc['estimated_full_minus_roi_with_selector_ms'] == 1.9


def test_break_even_false():
    calc = module.break_even_estimate(1.0, 2.0, 1.1)
    assert calc['potential_break_even'] is False
    assert calc['estimated_full_minus_roi_with_selector_ms'] == -2.1


def test_break_even_equal_does_not_claim_win():
    assert module.break_even_estimate(3.1, 2.0, 1.1)['potential_break_even'] is False


def test_break_even_rejects_negative():
    with pytest.raises(ValueError, match='negative'):
        module.break_even_estimate(1, 2, -1)


def test_load_frames_samples_deterministically(tmp_path):
    image_root = tmp_path / 'images'
    image_root.mkdir()
    for i in range(5):
        im = np.full((256, 320), i + 1, dtype=np.uint8)
        assert cv2.imwrite(str(image_root / f'frame_{i}.png'), im)
    conf = tmp_path / 'data.yaml'
    conf.write_text(yaml.safe_dump({'train_lr': str(image_root)}))
    a = module.load_frames(conf, 3, 5)
    b = module.load_frames(conf, 3, 5)
    assert len(a) == 3
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_load_frames_rejects_bad_shape(tmp_path):
    image_root = tmp_path / 'images'
    image_root.mkdir()
    for i in range(3):
        assert cv2.imwrite(str(image_root / f'frame_{i}.png'), np.zeros((32, 32), np.uint8))
    conf = tmp_path / 'data.yaml'
    conf.write_text(yaml.safe_dump({'train_lr': str(image_root)}))
    with pytest.raises(ValueError, match='Invalid 320x256'):
        module.load_frames(conf, 2, 42)


def test_cpu_network_output_shape():
    model = module.FSRCNN(scale=2, d=56, s=12, m=4).eval()
    with torch.inference_mode():
        actual = model(torch.zeros(1, 1, 16, 16))
    assert tuple(actual.shape) == (1, 1, 32, 32)


def test_trained_checkpoint_load_roundtrip_cpu(tmp_path):
    kw = {'scale': 2, 'd': 56, 's': 12, 'm': 4}
    model = module.FSRCNN(**kw)
    path = tmp_path / 'checkpoint.pt'
    torch.save({'state_dict': model.state_dict(), 'model_kwargs': kw}, path)
    loaded, saved_kw = module.load_trained_model(path, torch.device('cpu'))
    assert saved_kw == kw
    assert not loaded.training
