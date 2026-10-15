"""CPU-only unit tests; GPU smoke test must still run on the user's Docker/RTX GPU."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from check_large_fsrcnn_fullframe_equiv_step9f import compare_tile, summarize_rows, load_config


def test_identical():
    x = np.zeros((128,160), np.uint8)
    assert compare_tile(x, x) == {'max_difference':0,'mae':0.0,'different_pixels':0,'num_pixels':20480}


def test_one_intensity():
    x = np.zeros((2,2), np.uint8)
    y = x.copy(); y[0,0] = 1
    assert compare_tile(x,y) == {'max_difference':1,'mae':0.25,'different_pixels':1,'num_pixels':4}


@pytest.mark.parametrize('x,y', [
    (np.zeros((2, 2),np.float32), np.zeros((2,2),np.float32)),
    (np.zeros((2, 2),np.uint8),np.zeros((3,2),np.uint8)),
    (np.zeros((2, 2),np.uint8),np.zeros((2,2,1),np.uint8)),
    (np.zeros((0, 2),np.uint8),np.zeros((0,2),np.uint8)),
])
def test_input_check(x,y):
    with pytest.raises(ValueError): compare_tile(x,y)


def make_row(halo, max_diff=0, changed=0, pixels=4):
    return {'halo':halo,'max_difference':max_diff,'different_pixels':changed,
            'num_pixels':pixels,'abs_error_total':changed}


def test_summarize_good():
    s=summarize_rows([make_row(10,1,1),make_row(10),make_row(16)], [10,16],1)
    assert s['10']['images']==2
    assert s['10']['max_abs_pixel_difference_vs_full']==1
    assert s['10']['within_tolerance_on_all_images']
    assert s['10']['fraction_pixels_changed_vs_full']==0.125


def test_summarize_bad():
    s=summarize_rows([make_row(10,2,1)], [10],1)
    assert s['10']['images_exceeding_tolerance']==1
    assert not s['10']['within_tolerance_on_all_images']


def test_empty():
    with pytest.raises(ValueError): summarize_rows([], [10],1)


def test_missing_halo():
    with pytest.raises(ValueError): summarize_rows([make_row(12)], [10],1)


def test_config(tmp_path):
    cfg=tmp_path/'c.yaml'
    cfg.write_text('''selective_config: configs/a.yaml\nsr_checkpoint: models/sr.pt\npredictor_checkpoint: models/p.pt\npredictor_source_csv: reports/p.csv\nhalos: [10, 12, 16]\nlimit: 0\nmax_uint8_difference_tolerance: 1\n''')
    assert load_config(cfg)['halos']==[10,12,16]


def test_config_reject_duplicates(tmp_path):
    cfg=tmp_path/'c.yaml'
    cfg.write_text('''selective_config: configs/a.yaml\nsr_checkpoint: models/sr.pt\npredictor_checkpoint: models/p.pt\npredictor_source_csv: reports/p.csv\nhalos: [10, 10]\nlimit: 0\nmax_uint8_difference_tolerance: 1\n''')
    with pytest.raises(ValueError):load_config(cfg)
