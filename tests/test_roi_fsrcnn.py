"""Step 6C2: CPU-only unit tests, no dataset/weights/GPU needed."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from utitherm.sr.fsrcnn import FSRCNN
from utitherm.sr.roi_fsrcnn import (
    full_fsrcnn, fsrcnn_tile, render_fsrcnn, selected_region_difference,
)
from probe_roi_utility import roi_grid


def setup():
    torch.manual_seed(23)
    model = FSRCNN().eval()
    # Keep outputs nonzero; an untrained FSRCNN often quantizes everything to 0.
    with torch.no_grad():
        model.upsample.bias.fill_(0.2)
    rng = np.random.default_rng(42)
    im = rng.integers(10, 245, size=(256, 320), dtype=np.uint8)
    tiles = roi_grid(640, 512, 4, 4)
    return model, im, tiles


def test_full_shape_dtype():
    model, im, _ = setup()
    out = full_fsrcnn(im, model, torch.device('cpu'))
    assert out.shape == (512, 640) and out.dtype == np.uint8


@pytest.mark.parametrize('i', [0, 1, 5, 6, 10, 15])
def test_roi_crop_matches_full_sr_with_halo(i):
    model, im, tiles = setup()
    full = full_fsrcnn(im, model, torch.device('cpu'))
    roi, pixels = fsrcnn_tile(im, tiles[i], model, torch.device('cpu'), halo=16)
    x0, y0, x1, y1 = tiles[i]
    err = np.abs(roi.astype(np.int16) - full[y0:y1, x0:x1].astype(np.int16))
    assert err.max() <= 1, f'tile={i}, max={err.max()}'
    assert pixels < im.size


def test_selective_roi_equals_reference_after_blending():
    model, im, tiles = setup()
    chosen = [0, 5, 10, 15]
    full, nfull = render_fsrcnn(im, tiles, chosen, model, torch.device('cpu'), 'selective_fullframe')
    roi, ncrop = render_fsrcnn(im, tiles, chosen, model, torch.device('cpu'), 'selective_roi')
    diff = selected_region_difference(full, roi, tiles, chosen)
    assert diff['max_abs_difference'] <= 1
    assert ncrop < nfull


def test_selective_changes_only_selected_tiles():
    model, im, tiles = setup()
    out, _ = render_fsrcnn(im, tiles, [1], model, torch.device('cpu'), 'selective_roi')
    lin, _ = render_fsrcnn(im, tiles, [], model, torch.device('cpu'), 'linear')
    x0,y0,x1,y1 = tiles[1]
    mask=np.zeros(lin.shape,dtype=bool); mask[y0:y1,x0:x1]=True
    assert np.array_equal(out[~mask], lin[~mask])


def test_full_and_linear_modes_process_correct_input_pixels():
    model, im, tiles = setup()
    assert render_fsrcnn(im, tiles, [], model, torch.device('cpu'), 'linear')[1] == 0
    assert render_fsrcnn(im, tiles, [], model, torch.device('cpu'), 'full')[1] == im.size


@pytest.mark.parametrize('halo', [0, 2, 9, -1])
def test_halo_too_small_rejected(halo):
    model, im, tiles=setup()
    with pytest.raises(ValueError, match='halo'):
        fsrcnn_tile(im, tiles[0], model, torch.device('cpu'), halo)


def test_wrong_frame_and_invalid_index_rejected():
    model, im, tiles=setup()
    with pytest.raises(ValueError):
        render_fsrcnn(im.astype(np.float32), tiles, [0], model, torch.device('cpu'), 'selective_roi')
    with pytest.raises(ValueError):
        render_fsrcnn(im, tiles, [16], model, torch.device('cpu'), 'selective_roi')
    with pytest.raises(ValueError):
        render_fsrcnn(im, tiles, [0,0], model, torch.device('cpu'), 'selective_roi')


def test_comparison_detects_artifacts():
    _, _, tiles=setup()
    a=np.zeros((512,640),np.uint8)
    b=a.copy();b[3,4]=3
    diff=selected_region_difference(a,b,tiles,[0])
    assert diff['changed_output_pixels']==1
    assert diff['max_abs_difference']==3


def test_smoke_config_defaults():
    from benchmark_selective_fsrcnn import validate_settings
    import yaml
    cfg = yaml.safe_load((ROOT / 'configs' / 'selective_sr_eval.yaml').read_text()) if (ROOT / 'configs' / 'selective_sr_eval.yaml').exists() else {
        'train_lr':'data/processed/yolo_flir_lr_x2/images/train',
        'budget_tiles':4, 'imgsz':640, 'feather_hr_px':8,
        'confidence':0.25, 'nms_iou':0.7,
        'false_positive_penalty':0.25,'device':'0'
    }
    validate_settings(cfg)
    cfg=dict(cfg);cfg['imgsz']=320
    with pytest.raises(ValueError,match='640'):
        validate_settings(cfg)


def fake_benchmark_rows(images=3, repeats=3):
    from benchmark_selective_fsrcnn import MODES
    out = []
    for im in range(images):
        for rep in range(repeats):
            for mode in MODES:
                extra = {'linear_640': 0.0, 'full_bicubic_640': 0.3,
                         'full_fsrcnn_640': 4.0,
                         'selective_fullframe_fsrcnn4': 6.0,
                         'selective_roi_fsrcnn4': 3.0}[mode]
                out.append({'image': f'{im}.png', 'repeat': rep, 'method': mode,
                            'selector_ms': 2.0 if mode.startswith('selective') else 0.0,
                            'sr_or_resize_ms': 1 + extra, 'detector_ms': 10.0,
                            'total_ms': 11 + extra + 0.1*im + 0.02*rep,
                            'pytorch_peak_allocated_mib': 100.})
    return out


def test_benchmark_aggregates_by_image_not_repeat():
    from benchmark_selective_fsrcnn import collapse_and_compare
    r=collapse_and_compare(fake_benchmark_rows(), resamples=100)
    assert r['images'] == 3 and r['repeats'] == 3
    key='full_fsrcnn_640_minus_selective_roi_fsrcnn4_ms'
    assert r['paired_comparisons'][key]['mean_paired_difference_ms'] == pytest.approx(1.)


def test_duplicate_timing_trial_rejected():
    from benchmark_selective_fsrcnn import collapse_and_compare
    rows=fake_benchmark_rows()
    with pytest.raises(ValueError):
        collapse_and_compare(rows + [rows[0]], resamples=100)
