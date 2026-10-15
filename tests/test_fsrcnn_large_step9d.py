"""CPU-only guardrails for Step 9D; no FLIR dataset or CUDA needed."""
from pathlib import Path
import importlib.util
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

def import_from_file(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

prepare = import_from_file('prepare_fsrcnn_large_step9d_config', ROOT / 'scripts/prepare_fsrcnn_large_step9d_config.py')
train = import_from_file('train_fsrcnn_large_step9d', ROOT / 'scripts/train_fsrcnn_large_step9d.py')

@pytest.fixture
def base():
    return dict(train_hr='data/flir/images/train', output='reports/sr/fsrcnn_x2_v1',
                canonical_checkpoint='models/sr/fsrcnn_x2_v1.pt', epochs=30,
                batch=16, lr_patch=64, learning_rate=.001, seed=42,
                val_fraction=.1, workers=2, patience=8, d=56, s=12, m=4)


def test_parameters_match_benchmark():
    # Exact parameter count derived from current FSRCNN conv/prelu topology.
    d, s, m = 168, 36, 10
    assert 110*d + 2*d*s + 2*s + m*(9*s*s + 2*s) + 1 == 148009


def test_model_forward_cpu():
    from utitherm.sr.fsrcnn import FSRCNN
    model = FSRCNN(scale=2, d=168, s=36, m=10).eval()
    assert sum(p.numel() for p in model.parameters()) == 148009
    with torch.no_grad():
        assert tuple(model(torch.zeros(1, 1, 12, 16)).shape) == (1, 1, 24, 32)


def test_config_clones_protocol(base):
    cfg = prepare.make_large_config(base)
    train.validate_large_config(cfg, base)
    assert (cfg['d'],cfg['s'],cfg['m']) == (168,36,10)
    assert cfg['train_hr'] == base['train_hr']
    assert cfg['seed'] == base['seed']
    assert cfg['val_fraction'] == base['val_fraction']
    assert cfg['batch'] == 8
    assert cfg['output'] != base['output']
    assert cfg['canonical_checkpoint'] != base['canonical_checkpoint']
    assert base['d'] == 56


def test_small_baseline_batch_retained(base):
    base['batch'] = 4
    assert prepare.make_large_config(base)['batch'] == 4


def test_missing_base_key_fails(base):
    del base['seed']
    with pytest.raises(ValueError, match='missing'):
        prepare.make_large_config(base)


def test_nontraining_dataset_fails(base):
    base['train_hr'] = 'data/flir/images/val'
    with pytest.raises(ValueError, match='train_hr'):
        prepare.make_large_config(base)


def test_baseline_original_unchanged(base):
    original = base.copy()
    prepare.make_large_config(base)
    assert base == original


def test_wrong_large_architecture_fails(base):
    cfg = prepare.make_large_config(base)
    cfg['m'] = 9
    with pytest.raises(ValueError, match='d=168'):
        train.validate_large_config(cfg, base)


def test_edited_split_rejected(base):
    cfg = prepare.make_large_config(base)
    cfg['val_fraction'] = 0.2
    with pytest.raises(ValueError, match='val_fraction'):
        train.validate_large_config(cfg, base)


def test_same_checkpoint_rejected(base):
    cfg = prepare.make_large_config(base)
    cfg['canonical_checkpoint'] = base['canonical_checkpoint']
    with pytest.raises(ValueError, match='checkpoint'):
        train.validate_large_config(cfg, base)


def test_batch_safety(base):
    cfg = prepare.make_large_config(base)
    cfg['batch'] = 100
    with pytest.raises(ValueError, match='batch'):
        train.validate_large_config(cfg, base)


def test_exclusive_file_creation(tmp_path, base):
    import yaml
    path = tmp_path / 'original.yaml'
    path.write_text(yaml.safe_dump(base), encoding='utf-8')
    dst = tmp_path / 'large.yaml'
    output = prepare.prepare(path, dst)
    assert output['d'] == 168
    assert yaml.safe_load(dst.read_text(encoding='utf-8')) == output
    with pytest.raises(FileExistsError):
        prepare.prepare(path, dst)
