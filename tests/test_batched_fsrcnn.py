"""Step 6C3 batch-inference correctness checks: no FLIR data or GPU required."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

from utitherm.sr.fsrcnn import FSRCNN
from utitherm.sr.roi_fsrcnn import full_fsrcnn, render_fsrcnn
from utitherm.sr.roi_fsrcnn_batched import (
    roi_window, batched_fsrcnn_tiles, render_batched_fsrcnn,
)
from probe_roi_utility import roi_grid


@pytest.fixture(scope='module')
def sample():
    torch.manual_seed(23)
    torch.set_num_threads(1)
    net = FSRCNN().eval()
    with torch.no_grad():
        net.upsample.bias.fill_(0.4)
    gray = np.random.default_rng(94).integers(0, 256, (256, 320), dtype=np.uint8)
    tiles = roi_grid(640, 512, 4, 4)
    return net, gray, tiles, torch.device('cpu')


@pytest.mark.parametrize('i', [0, 1, 2, 3, 4, 5, 10, 12, 15])
def test_uniform_crop_includes_tile_and_stays_within_image(sample, i):
    _, _, tiles, _ = sample
    l, t, r, b = roi_window(tiles[i])
    assert (r-l, b-t) == (112, 96)
    assert 0 <= l < r <= 320 and 0 <= t < b <= 256
    x0, y0, x1, y1 = tiles[i]
    assert l <= x0//2 < x1//2 <= r
    assert t <= y0//2 < y1//2 <= b


def test_batch_one_pass_and_shapes(sample):
    net, gray, tiles, device = sample
    calls = []
    h = net.register_forward_pre_hook(lambda _m, inputs: calls.append(tuple(inputs[0].shape)))
    try:
        patches, processed = batched_fsrcnn_tiles(gray, tiles, [0, 5, 10, 15], net, device)
    finally:
        h.remove()
    assert calls == [(4, 1, 96, 112)], f'Expected one FSRCNN forward; got {calls}'
    assert list(patches) == [0, 5, 10, 15]
    assert all(p.shape == (128, 160) and p.dtype == np.uint8 for p in patches.values())
    assert processed == 4*112*96


@pytest.mark.parametrize('selected', ([0, 5, 10, 15], [3, 7, 11, 15], [0, 1, 2, 3]))
def test_batched_matches_sequential_and_full_reference(sample, selected):
    net, gray, tiles, device = sample
    full, _ = render_fsrcnn(gray, tiles, selected, net, device, 'selective_fullframe')
    serial, _ = render_fsrcnn(gray, tiles, selected, net, device, 'selective_roi')
    batched, _ = render_batched_fsrcnn(gray, tiles, selected, net, device)
    delta_full = np.abs(batched.astype(np.int16)-full.astype(np.int16))
    delta_serial = np.abs(batched.astype(np.int16)-serial.astype(np.int16))
    assert int(delta_full.max()) <= 1
    assert int(delta_serial.max()) <= 1


def test_only_selected_regions_changed(sample):
    net, gray, tiles, device = sample
    chosen=[1,6,10,15]
    output, count=render_batched_fsrcnn(gray, tiles, chosen, net, device)
    background, _=render_fsrcnn(gray, tiles, [], net, device, 'linear')
    mask=np.zeros(background.shape,dtype=bool)
    for i in chosen:
        x0,y0,x1,y1=tiles[i]
        mask[y0:y1,x0:x1]=True
    assert np.array_equal(output[~mask],background[~mask])
    assert count==4*96*112


@pytest.mark.parametrize('selected', ([],[0,0],[16],[-1]))
def test_reject_bad_selection(sample, selected):
    net, gray, tiles, device=sample
    with pytest.raises(ValueError):
        render_batched_fsrcnn(gray,tiles,selected,net,device)


def test_invalid_inputs_and_halo(sample):
    net, gray, tiles, device=sample
    with pytest.raises(ValueError):
        render_batched_fsrcnn(gray.astype(np.float32),tiles,[0],net,device)
    for bad_halo in (1,9,100):
        with pytest.raises(ValueError):
            roi_window(tiles[0],halo=bad_halo)
    with pytest.raises(ValueError):
        render_batched_fsrcnn(gray,tiles,[0],net,device,feather=-1)


def test_benchmark_modes_and_summary(sample):
    from benchmark_batched_fsrcnn import MODES, summarize_trials
    assert 'selective_batched_roi_fsrcnn4' in MODES
    rows=[]
    for image in ('A.png','B.png','C.png'):
        for rep in range(3):
            for j,method in enumerate(MODES):
                val=(15 if method=='selective_batched_roi_fsrcnn4' else 18) +j*.01
                rows.append({'image':image,'repeat':rep,'method':method,
                             'selector_ms':2.,'sr_or_resize_ms':3.,
                             'detector_ms':10.,'total_ms':val+rep*.1,
                             'model_input_lr_pixels':43008,
                             'pytorch_peak_allocated_mib':200.})
    result=summarize_trials(rows,bootstraps=100)
    assert result['images']==3 and result['repeats']==3
    diff=result['paired_comparisons']['selective_roi_fsrcnn4_minus_batched_ms']
    assert diff['mean_paired_difference_ms']>0
    with pytest.raises(ValueError):
        summarize_trials(rows + [rows[0]],bootstraps=100)
    with pytest.raises(ValueError):
        summarize_trials(rows[:-1],bootstraps=100)


def test_benchmark_config_compatible():
    from benchmark_batched_fsrcnn import validate_settings
    import yaml
    cfg=yaml.safe_load((ROOT/'configs/selective_sr_eval.yaml').read_text())
    validate_settings(cfg)
