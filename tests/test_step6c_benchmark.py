"""Step 6C1 CPU-only analysis tests; no real GPU or dataset needed."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_step6c import MODES, per_image_medians, percentile, summarize_benchmark, validate_config


def sample_rows(images=5, repeats=3):
    rows=[]
    for i in range(images):
        for r in range(repeats):
            for mode in MODES:
                latency={'linear_640':8.0,'full_bicubic_640':9.0,
                         'fullframe_selective4':14.0,'roi_only_selective4':12.0}[mode]
                rows.append({'image':f'frame{i:03d}.png','method':mode,'repeat':r,
                             'selector_ms':2 if 'selective' in mode else 0,
                             'prep_ms':1, 'detector_ms':latency-3,
                             'total_ms':latency + i*0.1 + r*.01})
    return rows


def test_per_image_medians_are_not_repeated_as_independent_frames():
    rows=sample_rows(5,3)
    med=per_image_medians(rows,'total_ms')
    assert len(med['roi_only_selective4'])==5
    assert med['roi_only_selective4']['frame000.png']==pytest.approx(12.01)


def test_report_keeps_all_four_methods():
    s=summarize_benchmark(sample_rows(),bootstrap=100)
    assert s['images']==5
    assert set(s['methods'])==set(MODES)
    assert s['methods']['roi_only_selective4']['total_ms']['median_of_image_medians']>0


def test_paired_mean_indicates_roi_only_faster_than_fullframe_but_slower_than_full_bicubic():
    s=summarize_benchmark(sample_rows(),bootstrap=100)
    dif=s['paired_comparisons']
    assert dif['fullframe_selective4_minus_roi_only_selective4_ms']['mean_paired_difference_ms']==2
    assert dif['full_bicubic_640_minus_roi_only_selective4_ms']['mean_paired_difference_ms']==-3


def test_quantiles_and_invalid_inputs():
    assert percentile([1,2,3],50)==2
    with pytest.raises(ValueError): percentile([],50)
    with pytest.raises(ValueError): percentile([1,2],101)
    with pytest.raises(ValueError): summarize_benchmark([],bootstrap=100)
    with pytest.raises(ValueError): summarize_benchmark(sample_rows(),bootstrap=50)


def test_missing_policy_rejected():
    rows=[x for x in sample_rows() if x['method']!='linear_640']
    with pytest.raises(ValueError, match='method'): summarize_benchmark(rows,bootstrap=100)


def test_missing_measurements_rejected():
    rows=sample_rows()
    rows.pop()
    with pytest.raises(ValueError,match='Incomplete'): summarize_benchmark(rows,bootstrap=100)


def test_config_rejects_wrong_domain():
    with pytest.raises(ValueError): validate_config({})
