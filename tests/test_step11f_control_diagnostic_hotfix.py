"""CPU-only regression tests for non-destructive Step11F reference audit."""
import csv
import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('step11f',(ROOT/'scripts/evaluate_bicubic_first_combined_step11f.py'))
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

CONTROLS=('full_bicubic_640','full_large_fsrcnn_640')
FIELDS=('small_gt','small_tp','all_tp','fp')
NAMES=('video-a-frame-001.png','video-b-frame-001.png')

def make_rows():
    rows=[]
    for im in NAMES:
        for mode in CONTROLS:
            rows.append({'image':im,'sequence':s.seq_of(im),'method':mode,
                         'small_gt':10,'small_tp':5+(mode==CONTROLS[1]),
                         'all_tp':7+(mode==CONTROLS[1]),'fp':3})
    return rows

def setup_ref(tmp_path, rows=None):
    rows=make_rows() if rows is None else rows
    path=tmp_path/'detection_per_image.csv'
    with path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ref={'results':{'methods':{mode:{k:sum(int(r[k]) for r in rows if r['method']==mode)
                                           for k in FIELDS} for mode in CONTROLS}}}
    return path,ref

def test_reference_agreement_passes(tmp_path):
    path,ref=setup_ref(tmp_path)
    result=s.reference_control_differences(make_rows(),path,ref)
    assert result['passed'] and result['different_images']==[]
    assert result['current_rows']==4 and result['reference_rows']==4
    assert all(v['mismatch_images']==0 for v in result['counts'].values())

def test_one_large_sr_tp_mismatch_is_recorded(tmp_path):
    path,ref=setup_ref(tmp_path)
    changed=make_rows()
    row=next(r for r in changed if r['image']==NAMES[0] and r['method']==CONTROLS[1])
    row['small_tp']+=1
    result=s.reference_control_differences(changed,path,ref)
    assert result['passed'] is False
    assert result['counts']['full_bicubic_640']['mismatch_images']==0
    assert result['counts']['full_large_fsrcnn_640']['delta_totals']['small_tp']==1
    assert result['different_images']==[{
        'image':NAMES[0],'sequence':s.seq_of(NAMES[0]),'method':'full_large_fsrcnn_640',
        'ref_small_gt':10,'ref_small_tp':6,'ref_all_tp':8,'ref_fp':3,
        'current_small_gt':10,'current_small_tp':7,'current_all_tp':8,'current_fp':3,
        'delta_small_gt':0,'delta_small_tp':1,'delta_all_tp':0,'delta_fp':0
    }]

def test_aggregate_cancellation_is_still_failure(tmp_path):
    path,ref=setup_ref(tmp_path)
    changed=make_rows()
    for row in changed:
        if row['method']==CONTROLS[1]:
            row['small_tp']+=1 if row['image']==NAMES[0] else -1
    result=s.reference_control_differences(changed,path,ref)
    assert not result['passed']
    assert result['counts'][CONTROLS[1]]['mismatch_images']==2
    assert result['counts'][CONTROLS[1]]['delta_totals']['small_tp']==0

def test_reference_csv_summary_disagreement_rejected(tmp_path):
    path,ref=setup_ref(tmp_path)
    ref['results']['methods'][CONTROLS[1]]['fp']+=1
    with pytest.raises(ValueError,match='disagrees with its summary'):
        s.reference_control_differences(make_rows(),path,ref)

def test_missing_reference_row_rejected(tmp_path):
    path,ref=setup_ref(tmp_path)
    with pytest.raises(ValueError,match='control images differ'):
        s.reference_control_differences(make_rows()[:-1],path,ref)

def test_duplicate_reference_row_rejected(tmp_path):
    rows=make_rows();path,ref=setup_ref(tmp_path,rows+[rows[0]])
    with pytest.raises(ValueError,match='Duplicate reference'):
        s.reference_control_differences(make_rows(),path,ref)

def test_missing_csv_rejected(tmp_path):
    _,ref=setup_ref(tmp_path)
    with pytest.raises(FileNotFoundError):
        s.reference_control_differences(make_rows(),tmp_path/'absent.csv',ref)

def test_corrupt_sequence_rejected(tmp_path):
    path,ref=setup_ref(tmp_path)
    changed=make_rows();changed[0]['sequence']='wrong'
    with pytest.raises(ValueError,match='Invalid current sequence'):
        s.reference_control_differences(changed,path,ref)

def test_silent_reference_override_not_added():
    import inspect
    source=inspect.getsource(s.run)
    assert 'verify_reference(stats,ref)' in source
    assert 'if failure:' in source
    assert "raise ValueError(f'{failure}; see saved Step11F reference diagnostics." in source
