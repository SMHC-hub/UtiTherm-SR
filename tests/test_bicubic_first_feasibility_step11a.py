"""Step 11A CPU protocol tests; real detector/GPU validation requires smoke run."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from utitherm.sr.bicubic_first_fsrcnn_step11a import (  # noqa: E402
    composite_bicubic_patches, render_bicubic_first_fsrcnn)
from utitherm.sr.roi_bicubic import _blend  # noqa: E402

CFG = ROOT / 'configs' / 'bicubic_first_feasibility_step11a.yaml'
SCRIPT = ROOT / 'scripts' / 'collect_bicubic_first_feasibility_step11a.py'
TILES = [(x * 160, y * 128, (x + 1) * 160, (y + 1) * 128)
         for y in range(4) for x in range(4)]


class NearestX2(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')


def sample(seed=5):
    return np.random.default_rng(seed).integers(0, 256, (256, 320), dtype=np.uint8)


def mask(selected):
    m = np.zeros((512, 640), dtype=bool)
    for idx in selected:
        x0, y0, x1, y1 = TILES[idx]
        m[y0:y1, x0:x1] = True
    return m


@pytest.mark.parametrize('selected', [[0], [5], [15], [0, 5, 10, 15]])
def test_renderer_unchanged_outside_neural_roi(selected):
    lr = sample(3)
    baseline = cv2.resize(lr, (640, 512), interpolation=cv2.INTER_CUBIC)
    out, processed = render_bicubic_first_fsrcnn(lr, TILES, selected, NearestX2(), torch.device('cpu'))
    assert np.array_equal(out[~mask(selected)], baseline[~mask(selected)])
    assert np.array_equal(lr, sample(3))
    assert processed == 8400 * len(selected)


@pytest.mark.parametrize('selected', [[0], [6], [15], [0, 5, 10, 15]])
def test_renderer_matches_direct_patch_blend(selected):
    from utitherm.sr.roi_fsrcnn_batched import batched_fsrcnn_tiles
    lr = sample(9)
    cubic = cv2.resize(lr, (640, 512), interpolation=cv2.INTER_CUBIC)
    patches, _ = batched_fsrcnn_tiles(lr, TILES, selected, NearestX2(), torch.device('cpu'), halo=10)
    expected = cubic.copy()
    for i in selected:
        _blend(expected, patches[i], TILES[i], 8)
    actual, _ = render_bicubic_first_fsrcnn(lr, TILES, selected,
                                             NearestX2(), torch.device('cpu'), bicubic=cubic)
    assert np.array_equal(actual, expected)
    assert np.array_equal(cubic, cv2.resize(lr, (640, 512), interpolation=cv2.INTER_CUBIC))


def test_bicubic_first_differs_from_linear_background():
    from utitherm.sr.roi_fsrcnn_batched import render_batched_fsrcnn
    lr = sample(12)
    new, _ = render_bicubic_first_fsrcnn(lr, TILES, [5], NearestX2(), torch.device('cpu'))
    old, _ = render_batched_fsrcnn(lr, TILES, [5], NearestX2(), torch.device('cpu'), halo=10)
    outside = ~mask([5])
    assert np.any(new[outside] != old[outside])


def test_selected_region_contains_real_neural_change():
    lr = sample(29)
    cubic = cv2.resize(lr, (640, 512), interpolation=cv2.INTER_CUBIC)
    neural, _ = render_bicubic_first_fsrcnn(lr, TILES, [6], NearestX2(), torch.device('cpu'))
    assert np.any(neural[mask([6])] != cubic[mask([6])])


@pytest.mark.parametrize('bad', [[], [1, 1], [16], [-1]])
def test_bad_tile_selection_rejected(bad):
    lr = sample()
    with pytest.raises(ValueError):
        render_bicubic_first_fsrcnn(lr, TILES, bad, NearestX2(), torch.device('cpu'))


@pytest.mark.parametrize('arg,val', [('feather', 4), ('halo', 16)])
def test_protocol_halo_feather_frozen(arg, val):
    kwargs = {arg: val}
    with pytest.raises(ValueError):
        render_bicubic_first_fsrcnn(sample(), TILES, [0], NearestX2(), torch.device('cpu'), **kwargs)


def test_bicubic_input_cannot_be_modified_by_compositor():
    bicubic = cv2.resize(sample(8), (640, 512), interpolation=cv2.INTER_CUBIC)
    before = bicubic.copy()
    patch = np.full((128, 160), 255, np.uint8)
    new = composite_bicubic_patches(bicubic, {0: patch}, TILES, [0])
    assert np.array_equal(bicubic, before)
    assert np.array_equal(new[~mask([0])], bicubic[~mask([0])])


@pytest.mark.parametrize('bad', [np.zeros((512, 640), np.float32), np.zeros((10, 10), np.uint8)])
def test_bad_bicubic_rejected(bad):
    with pytest.raises(ValueError):
        composite_bicubic_patches(bad, {0: np.zeros((128, 160), np.uint8)}, TILES, [0])


def test_wrong_neural_patch_shape_rejected():
    with pytest.raises(ValueError):
        composite_bicubic_patches(np.zeros((512, 640), np.uint8),
                                  {0: np.zeros((100, 100), np.uint8)}, TILES, [0])


def test_patch_selection_mismatch_rejected():
    with pytest.raises(ValueError):
        composite_bicubic_patches(np.zeros((512, 640), np.uint8),
                                  {1: np.zeros((128, 160), np.uint8)}, TILES, [0])


def _mod_stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


@pytest.fixture
def mod(monkeypatch):
    import hashlib
    def _sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def _seq(name):
        return name.split('-frame-', 1)[0]
    probe = _mod_stub('probe_roi_utility', roi_grid=lambda w, h, r, c: TILES,
                      detection_counts=lambda g,p,t: p,
                      prediction_records=lambda d, f, c: f,
                      utility=lambda a,b,k: {
                          'delta_small_tp': b['small_tp'] - a['small_tp'],
                          'delta_all_tp': b['all_tp'] - a['all_tp'],
                          'delta_fp': b['fp'] - a['fp'],
                          'utility_score': (b['small_tp'] - a['small_tp']) - k * (b['fp'] - a['fp']),
                          'positive_label': int((b['small_tp'] - a['small_tp']) - k * (b['fp'] - a['fp']) > 0),
                      })
    stubs = {
        'prepare_utility_v2_manifest': _mod_stub('prepare_utility_v2_manifest',
                            verify_manifest=lambda m,c: [], sha256=_sha, sequence_id=_seq),
        'probe_roi_utility': probe,
        'evaluate_bicubic_baseline': _mod_stub('evaluate_bicubic_baseline', ground_truth=lambda p: []),
        'benchmark_selective_fsrcnn': _mod_stub('benchmark_selective_fsrcnn', load_sr_checkpoint=lambda p,d: None),
    }
    for k,v in stubs.items():
        monkeypatch.setitem(sys.modules,k,v)
    spec = importlib.util.spec_from_file_location('isolated_step11a', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exp():
    return dict(halo=10, budget_tiles=1, feather_hr_px=8,
                baseline='full_bicubic_640',
                intervention='bicubic_base_single_large_fsrcnn_roi_x2_h10_feather8',
                expected_images=300, expected_sequences=33)


def base():
    return dict(grid_rows=4, grid_cols=4, imgsz=640, confidence=.25,
                nms_iou=.7, false_positive_penalty=.25, feather_hr_px=8, device='0')


def frozen():
    return dict(num_images=300, unique_sequences=33)


def test_config_file_available():
    assert CFG.is_file()
    import yaml
    assert yaml.safe_load(CFG.read_text()) == exp() | {
        'base_collection_config': 'configs/utility_v2_data.yaml',
        'frozen_manifest': 'reports/utility/v2_fresh_split_300_v2/manifest.json',
        'large_sr_checkpoint': 'models/sr/fsrcnn_large_x2_stable_step9d3_v1.pt',
    }


def test_config_valid(mod):
    mod.validate_config(exp(), base(), frozen())


@pytest.mark.parametrize('key,value', [('halo', 16),('budget_tiles', 4),('baseline','linear_640'),
                                      ('feather_hr_px',0),('intervention','bicubic_only')])
def test_invalid_interventions_rejected(mod, key, value):
    e = exp(); e[key] = value
    with pytest.raises(ValueError):
        mod.validate_config(e, base(), frozen())


@pytest.mark.parametrize('key,value', [('imgsz',320),('confidence',.3),('nms_iou',.6),
                                      ('false_positive_penalty',.5),('feather_hr_px',0),
                                      ('grid_cols',3),('device','cpu')])
def test_base_settings_frozen(mod, key, value):
    b = base(); b[key] = value
    with pytest.raises(ValueError):
        mod.validate_config(exp(),b,frozen())


def test_wrong_image_count_rejected(mod):
    f = frozen(); f['num_images'] = 100
    with pytest.raises(ValueError):
        mod.validate_config(exp(),base(),f)


def _shard(mod):
    name = 'video-ABCDE-frame-000001-xyz.png'
    baseline = {'small_tp': 4, 'all_tp': 6, 'fp': 3, 'small_gt': 9}
    rows = []
    for i, rect in enumerate(TILES):
        rows.append({'image': name, 'sequence': 'video-ABCDE', 'tile_index': i,
                     **dict(zip(('tile_x0_hr','tile_y0_hr','tile_x1_hr','tile_y1_hr'),rect)),
                     'baseline_kind':'full_bicubic_640', 'processed_lr_pixels':8400,
                     'baseline_small_tp':4,'baseline_all_tp':6,'baseline_fp':3,
                     'candidate_small_tp':4,'candidate_all_tp':6,'candidate_fp':3,
                     'delta_small_tp':0,'delta_all_tp':0,'delta_fp':0,'utility_score':0.0,'positive_label':0})
    return {'image':name, 'sequence':'video-ABCDE', 'baseline':baseline, 'rows':rows}


def test_valid_shard(mod):
    s = _shard(mod)
    assert mod.shard_is_valid(s,s['image'])


@pytest.mark.parametrize('field,value', [('baseline_kind','linear_640'), ('utility_score',1),
                                         ('positive_label',1),('processed_lr_pixels',20),
                                         ('candidate_small_tp', 5), ('candidate_all_tp',7),
                                         ('candidate_fp',4),('tile_x0_hr',99)])
def test_corrupt_shard_rejected(mod, field, value):
    s = _shard(mod); s['rows'][0][field]=value
    assert not mod.shard_is_valid(s,s['image'])


def test_missing_shard_tile_rejected(mod):
    s = _shard(mod); s['rows'].pop()
    assert not mod.shard_is_valid(s,s['image'])


def test_duplicate_shard_tile_rejected(mod):
    s = _shard(mod); s['rows'][0]['tile_index']=1
    assert not mod.shard_is_valid(s,s['image'])


def test_wrong_shard_filename_rejected(mod):
    s = _shard(mod)
    assert not mod.shard_is_valid(s,'video-FGHI-frame-000001-xyz.png')


def test_make_rows_bicubic_baseline_exact(mod, monkeypatch):
    img = sample(14)
    cubic = cv2.resize(img,(640,512),interpolation=cv2.INTER_CUBIC)
    linear = cv2.resize(img,(640,512),interpolation=cv2.INTER_LINEAR)
    assert not np.array_equal(cubic,linear)
    seen = []
    cfg = {'confidence':.25,'false_positive_penalty':.25,'feather_hr_px':8}
    baseline = {'small_tp':4,'all_tp':6,'fp':3,'small_gt':9}
    changed = {'small_tp':5,'all_tp':7,'fp':4,'small_gt':9}
    monkeypatch.setattr(mod,'prediction_records',lambda detector, frame, cfg: frame)
    monkeypatch.setattr(mod,'detection_counts', lambda gts,frame,t: baseline if np.array_equal(frame,cubic) else changed)
    def fake_renderer(lr,tiles,selected,sr,device,feather,halo,bicubic):
        assert np.array_equal(bicubic,cubic)
        seen.append(selected[0])
        candidate = cubic.copy()
        candidate[32,32] = (int(candidate[32,32])+1)%256
        return candidate,8400
    monkeypatch.setattr(mod,'render_bicubic_first_fsrcnn', fake_renderer)
    a, rows = mod.make_rows(img,'video-ABCDE-frame-000001-xyz.png',[],None,None,None,cfg,TILES)
    assert a==baseline
    assert seen==list(range(16))
    assert len(rows)==16 and all(r['delta_small_tp']==1 and r['delta_fp']==1 for r in rows)
    assert all(r['utility_score']==.75 for r in rows)
    assert mod.shard_is_valid({'image':rows[0]['image'], 'sequence':'video-ABCDE','baseline':baseline,'rows':rows},rows[0]['image'])


def test_csv_atomic(mod,tmp_path):
    s=_shard(mod)
    csv_file=tmp_path/'table.csv'
    mod.save_csv(csv_file,s['rows'],mod.COLUMNS)
    import csv
    assert len(list(csv.DictReader(csv_file.open())))==16
    assert not (tmp_path/'table.csv.tmp').exists()


def test_provenance_hashes(mod,tmp_path):
    files=[]
    for k in range(5):
        p=tmp_path/f'{k}.txt';p.write_text(str(k)); files.append(p)
    manifest=mod.provenance(*files,['video-ABCDE-frame-000001-xyz.png'],exp())
    assert manifest['baseline']=='full_bicubic_640'
    assert manifest['sr_checkpoint_sha256']==mod.sha256(files[4])
    assert manifest['num_tiles']==16


def test_summary_oracle_marked_proxy(mod, tmp_path):
    s=_shard(mod)
    s['rows'][0].update(candidate_small_tp=5,delta_small_tp=1,utility_score=1.0,positive_label=1)
    s['rows'][1].update(candidate_fp=2,delta_fp=-1,utility_score=.25,positive_label=1)
    out=tmp_path
    (out/'incremental_roi_utility.csv').write_text('x')
    res=mod.summarize(s['rows'],[s],out,{'baseline':'full_bicubic_640'})
    assert res['positive_utility']==2
    assert res['images_with_any_positive']==1
    assert res['mean_top4_single_roi_oracle_proxy']==1.25
    assert any('NOT' in x for x in res['limitations'])
