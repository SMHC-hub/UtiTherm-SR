"""Fast CPU-only protocol tests; they do NOT replace GPU integration validation."""
from __future__ import annotations
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'collect_large_fsrcnn_utility_step10a.py'
CFG = Path(__file__).resolve().parents[1] / 'configs' / 'large_fsrcnn_utility_step10a.yaml'


def _stub(name, **things):
    m = types.ModuleType(name)
    for k, v in things.items():
        setattr(m, k, v)
    return m


@pytest.fixture
def mod(monkeypatch):
    sequence_id = lambda name: name.split('-frame-', 1)[0]
    import hashlib
    sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    stub_utils = _stub('prepare_utility_v2_manifest', verify_manifest=lambda m, c: [], sha256=sha256, sequence_id=sequence_id)
    stub_probe = _stub('probe_roi_utility', detection_counts=None,
                       prediction_records=None, roi_grid=lambda w, h, r, c: [(0, 0, 160, 128)] * 16,
                       utility=lambda a, b, penalty: {
                           'delta_small_tp': b['small_tp'] - a['small_tp'],
                           'delta_all_tp': b['all_tp'] - a['all_tp'],
                           'delta_fp': b['fp'] - a['fp'],
                           'utility_score': (b['small_tp'] - a['small_tp']) - penalty * (b['fp'] - a['fp']),
                           'positive_label': int((b['small_tp'] - a['small_tp']) - penalty * (b['fp'] - a['fp']) > 0),
                       })
    stubs = {
        'prepare_utility_v2_manifest': stub_utils,
        'probe_roi_utility': stub_probe,
        'evaluate_bicubic_baseline': _stub('evaluate_bicubic_baseline', ground_truth=lambda p: []),
        'benchmark_selective_fsrcnn': _stub('benchmark_selective_fsrcnn', load_sr_checkpoint=lambda p, device: None),
        'utitherm': _stub('utitherm'),
        'utitherm.sr': _stub('utitherm.sr'),
        'utitherm.sr.roi_fsrcnn_batched': _stub('utitherm.sr.roi_fsrcnn_batched', render_batched_fsrcnn=None),
    }
    for name, value in stubs.items():
        monkeypatch.setitem(sys.modules, name, value)
    spec = importlib.util.spec_from_file_location('temp_step10a', SCRIPT)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def _exp():
    return dict(halo=10, budget_tiles=1, intervention='large_fsrcnn_x2_roi_h10_linear_background_feather8',
                expected_images=300, expected_sequences=33)


def _base():
    return dict(grid_rows=4, grid_cols=4, imgsz=640, confidence=.25, nms_iou=.7,
                false_positive_penalty=.25, feather_hr_px=8, device='0')


def _frozen():
    return dict(num_images=300, unique_sequences=33)


def _shard(mod):
    name = 'video-ABCDE-frame-000001-random.png'
    base = dict(small_tp=4, all_tp=5, fp=3, small_gt=6)
    rows = []
    for i in range(16):
        row = dict(image=name, sequence='video-ABCDE', tile_index=i,
                   tile_x0_hr=0, tile_y0_hr=0, tile_x1_hr=160, tile_y1_hr=128,
                   enhanced_fraction=.0625, baseline_small_tp=4, baseline_fp=3,
                   candidate_small_tp=4, candidate_fp=3, delta_small_tp=0,
                   delta_all_tp=0, delta_fp=0, utility_score=0.0, positive_label=0)
        rows.append(row)
    return dict(image=name, sequence='video-ABCDE', baseline=base, rows=rows)


def test_shipped_config_exists():
    assert CFG.is_file()


def test_schema_string(mod):
    assert 'LARGE_FSRCNN' in mod.SCHEMA


def test_protocol_accepts_expected(mod):
    mod.validate_config(_exp(), _base(), _frozen())

@pytest.mark.parametrize('key,value', [
    ('halo', 12), ('budget_tiles', 4), ('intervention', 'bicubic'),
])
def test_refuse_changed_intervention(mod, key, value):
    e = _exp(); e[key] = value
    with pytest.raises(ValueError):
        mod.validate_config(e, _base(), _frozen())

@pytest.mark.parametrize('key,value', [
    ('confidence', 0.3), ('imgsz', 320), ('nms_iou', 0.6),
    ('false_positive_penalty', 0.5), ('feather_hr_px', 0), ('device', 'cpu'),
    ('grid_rows', 2),
])
def test_refuse_changed_baseline_settings(mod, key, value):
    b = _base(); b[key] = value
    with pytest.raises(ValueError):
        mod.validate_config(_exp(), b, _frozen())


def test_refuse_changed_image_count(mod):
    m = _frozen(); m['num_images'] = 299
    with pytest.raises(ValueError): mod.validate_config(_exp(), _base(), m)


def test_refuse_changed_sequence_count(mod):
    m = _frozen(); m['unique_sequences'] = 22
    with pytest.raises(ValueError): mod.validate_config(_exp(), _base(), m)


def test_valid_shard(mod):
    s = _shard(mod)
    assert mod.shard_is_valid(s, s['image'])


def test_shard_reject_missing_row(mod):
    s = _shard(mod); s['rows'].pop()
    assert not mod.shard_is_valid(s, s['image'])


def test_shard_reject_duplicate_tile(mod):
    s = _shard(mod); s['rows'][1]['tile_index'] = 0
    assert not mod.shard_is_valid(s, s['image'])


def test_shard_reject_wrong_image(mod):
    s = _shard(mod); s['rows'][1]['image'] = 'other.png'
    assert not mod.shard_is_valid(s, s['image'])


def test_shard_reject_inconsistent_score(mod):
    s = _shard(mod); s['rows'][1]['utility_score'] = 1.1
    assert not mod.shard_is_valid(s, s['image'])


def test_shard_reject_inconsistent_counts(mod):
    s = _shard(mod); s['rows'][3]['candidate_small_tp'] = 5
    assert not mod.shard_is_valid(s, s['image'])


def test_shard_reject_false_positive_label(mod):
    s = _shard(mod); s['rows'][3]['positive_label'] = 1
    assert not mod.shard_is_valid(s, s['image'])


def test_shard_reject_wrong_sequence(mod):
    s = _shard(mod); s['sequence'] = 'video-WRONG'
    assert not mod.shard_is_valid(s, s['image'])


def test_make_rows_calls_single_tile_render(mod, monkeypatch):
    base = dict(small_tp=5, all_tp=6, fp=4, small_gt=9)
    counter = []
    monkeypatch.setattr(mod, 'prediction_records', lambda detector, frame, cfg: frame)
    monkeypatch.setattr(mod, 'detection_counts', lambda gts, preds, threshold: base if isinstance(preds, np.ndarray) and np.array_equal(preds, mod.cv2.resize(image, (640,512), interpolation=mod.cv2.INTER_LINEAR)) else dict(small_tp=6, all_tp=7, fp=4, small_gt=9))
    def renderer(gray, tiles, selected, sr, device, feather, halo):
        counter.append((selected, feather, halo))
        return np.full((512, 640), 77, np.uint8), 100
    monkeypatch.setattr(mod, 'render_batched_fsrcnn', renderer)
    global image
    image = np.zeros((256, 320), np.uint8)
    name = 'video-ABCDE-frame-000001-random.png'
    rects = [(0, 0, 160, 128)] * 16
    baseline, rows = mod.make_rows(image, name, [], None, None, None,
                                   {'confidence': .25, 'feather_hr_px': 8, 'false_positive_penalty': .25}, rects, 10)
    assert baseline['small_tp'] == 5
    assert len(rows) == 16
    assert counter == [([i], 8, 10) for i in range(16)]
    assert all(r['delta_small_tp'] == 1 for r in rows)
    assert mod.shard_is_valid({'image':name,'sequence':'video-ABCDE','baseline':base,'rows':rows}, name)


def test_atomic_csv(mod, tmp_path):
    s = _shard(mod)
    path = tmp_path / 'roi.csv'
    mod.save_csv(path, s['rows'], mod.COLUMNS)
    import csv
    with path.open() as f:
        assert len(list(csv.DictReader(f))) == 16
    assert not (tmp_path / 'roi.csv.tmp').exists()


def test_source_record_hashes(mod, tmp_path):
    files = []
    for idx in range(5):
        p = tmp_path / str(idx)
        p.write_text(str(idx))
        files.append(p)
    rec = mod.provenance(*files, ['video-ABCDE-frame-000001-random.png'], _exp())
    assert rec['num_tiles'] == 16
    assert rec['sr_checkpoint_sha256'] == mod.sha256(files[4])
    assert rec['protocol'] == mod.SCHEMA
