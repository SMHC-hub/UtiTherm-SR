"""Step 6B CPU-only tests: pixel equivalence & ROI compute accounting."""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from utitherm.sr.roi_bicubic import cubic_tile_from_lr,render_selected
from probe_roi_utility import roi_grid
from evaluate_selective_sr import apply_tiles
from benchmark_roi_bicubic import MODES, make_frame, cpu_prep_bench, summarize

TILES=roi_grid(640,512,4,4)


def image(seed):
    return np.random.default_rng(seed).integers(0,256,size=(256,320),dtype=np.uint8)


@pytest.mark.parametrize('seed',[0,1,11])
def test_all_16_roi_cubic_patches_exact_full_frame(seed):
    lr=image(seed)
    full=cv2.resize(lr,(640,512),interpolation=cv2.INTER_CUBIC)
    for tile in TILES:
        x0,y0,x1,y1=tile
        patch,count=cubic_tile_from_lr(lr,tile,halo=2)
        assert count>patch.size
        assert np.array_equal(patch,full[y0:y1,x0:x1])


@pytest.mark.parametrize('chosen',[[0,1,2,3],[0,5,10,15],[1,4,11,13]])
def test_roi_exactly_matches_existing_step6a_intervention(chosen):
    lr=image(31)
    cropped,pixels=render_selected(lr,TILES,chosen,8,2,True)
    base=cv2.resize(lr,(640,512),interpolation=cv2.INTER_LINEAR)
    cubic=cv2.resize(lr,(640,512),interpolation=cv2.INTER_CUBIC)
    old=apply_tiles(base,cubic,TILES,chosen,8)
    full,newpixels=render_selected(lr,TILES,chosen,8,2,False)
    assert np.array_equal(cropped,old)
    assert np.array_equal(cropped,full)
    assert pixels<(640*512)
    assert newpixels==(640*512)


def test_roi_blending_changes_only_selected_region_and_never_modifies_source():
    lr=image(5); original=lr.copy()
    selected=[6,7,8,9]
    out,_=render_selected(lr,TILES,selected,8,2,True)
    base=cv2.resize(lr,(640,512),interpolation=cv2.INTER_LINEAR)
    mask=np.zeros(base.shape,dtype=bool)
    for idx in selected:
        x0,y0,x1,y1=TILES[idx]
        mask[y0:y1,x0:x1]=True
    assert np.array_equal(out[~mask],base[~mask])
    assert np.array_equal(lr,original)


def test_feather_zero_still_matches_reference():
    lr=image(6)
    out,_=render_selected(lr,TILES,[0,5,10,15],feather=0)
    base=cv2.resize(lr,(640,512),interpolation=cv2.INTER_LINEAR)
    cubic=cv2.resize(lr,(640,512),interpolation=cv2.INTER_CUBIC)
    assert np.array_equal(out,apply_tiles(base,cubic,TILES,[0,5,10,15],0))


@pytest.mark.parametrize('tiles,indices',[(TILES,[0,0]),(TILES,[16]),(TILES,[]),([(0,0,160,128),(0,0,160,128)],[0,1])])
def test_reject_invalid_selection(tiles,indices):
    with pytest.raises(ValueError):
        render_selected(image(0),tiles,indices)


def test_reject_inadequate_halo_unaligned_or_bad_dimensions():
    with pytest.raises(ValueError,match='halo'):
        cubic_tile_from_lr(image(1),TILES[0],halo=1)
    with pytest.raises(ValueError,match='align'):
        cubic_tile_from_lr(image(1),(1,0,160,128),halo=2)
    with pytest.raises(ValueError):
        render_selected(np.zeros((24,24),dtype=np.uint8),TILES,[0])


def test_interpolated_cubic_pixels_are_not_equal_to_enhanced_area():
    cropped,pixels=render_selected(image(3),TILES,[0,5,10,15],8,2,True)
    assert cropped.shape==(512,640)
    assert .25 < pixels/(640*512) < 1


def test_bench_has_all_mode_keys_and_positive_timings():
    timings=cpu_prep_bench(image(1),TILES,[0,5,10,15],MODES,8,2,3,42)
    assert set(timings)==set(MODES)
    assert all(v>0 for v in timings.values())


def test_make_frame_reference_modes():
    gray=image(2)
    linear,p0=make_frame(gray,TILES,[0,1,2,3],'linear_640',8,2)
    cubic,p1=make_frame(gray,TILES,[0,1,2,3],'full_bicubic_640',8,2)
    a,_=make_frame(gray,TILES,[0,1,2,3],'fullframe_selective4',8,2)
    b,_=make_frame(gray,TILES,[0,1,2,3],'roi_only_selective4',8,2)
    assert p0==0 and p1==640*512
    assert np.array_equal(a,b)
    assert not np.array_equal(linear,cubic)
    with pytest.raises(ValueError):
        make_frame(gray,TILES,[0,1,2,3],'bad',8,2)


def test_summary_aggregates_detected_objects_without_proxy():
    rows=[]
    for mode in MODES:
        rows.append({'method':mode,'cpu_prep_ms':1,'detect_ms':3,'selector_ms':.5,'measured_pipeline_ms':4.5,
                     'cubic_pixels':81920,'small_tp':5,'small_gt':10,'fp':2})
    summ=summarize(rows)
    assert len(summ)==4
    assert summ['roi_only_selective4']['small_recall']==.5
    assert summ['roi_only_selective4']['false_positives']==2
