from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_utility_predictor import (
    MODES, read_csv, sequence_id, audit_detection, paired_deltas,
    sequence_bootstrap_ci, summarise_pair, label_profile, run
)


def dataset(tmp_path):
    report=tmp_path/'report'; report.mkdir()
    mani=tmp_path/'manifest.json'
    ims=['video-A-frame-000001-foo.png','video-B-frame-000001-bar.png']
    mani.write_text(json.dumps({'images':ims}), encoding='utf-8')
    import hashlib
    summary={'protocol':'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST',
             'split_manifest_sha256':hashlib.sha256(mani.read_bytes()).hexdigest(),
             'metrics':{'images':2,'unique_sequences':2,'methods':{}}}
    vals={m:{'small_gt':10,'small_tp':5,'all_tp':6,'fp':2} for m in MODES}
    vals['predictor_bicubic4']['small_tp']=7;vals['predictor_bicubic4']['all_tp']=8;vals['predictor_bicubic4']['fp']=3
    for m in MODES:
        summary['metrics']['methods'][m]={k:v*2 for k,v in vals[m].items()}
    (report/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    rows=[]
    for im in ims:
        for m in MODES:
            rows.append({'image':im,'sequence':sequence_id(im),'method':m, **vals[m]})
    with (report/'detection_per_image.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    roi=tmp_path/'roi.csv'
    rr=[]
    for im in ims:
        for t in range(16):
            rr.append({'image':im,'sequence':sequence_id(im),'tile_index':t,'delta_small_tp':int(t==0),
                       'delta_fp':-1 if t==1 else 0,'utility_score':1 if t==0 else .25 if t==1 else 0,
                       'positive_label':int(t in (0,1))})
    with roi.open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    return report,mani,roi


def test_sequence_id():
    assert sequence_id('video-abc-frame-01-xyz.png')=='video-abc'


def test_reject_unsafe_name():
    with pytest.raises(ValueError): sequence_id('../video-abc-frame-01-xyz.png')


def test_audit_and_paired(tmp_path):
    report,mani,_=dataset(tmp_path)
    summary=json.loads((report/'summary.json').read_text())
    manifest=json.loads(mani.read_text());manifest['_path']=str(mani)
    rows=read_csv(report/'detection_per_image.csv',{'image','method','small_gt'})
    audited=audit_detection(rows,summary,manifest)
    changes=paired_deltas(audited,'predictor_bicubic4','random_bicubic4',.25)
    assert len(changes)==2 and changes[0]['delta_utility']==1.75


def test_reject_summary_mismatch(tmp_path):
    report,mani,_=dataset(tmp_path)
    summary=json.loads((report/'summary.json').read_text())
    summary['metrics']['methods']['linear_640']['fp']=999
    manifest=json.loads(mani.read_text());manifest['_path']=str(mani)
    with pytest.raises(ValueError,match='CSV/report mismatch'):
        audit_detection(read_csv(report/'detection_per_image.csv',{'image','method','small_gt'}),summary,manifest)


def test_reject_manifest_mismatch(tmp_path):
    report,mani,_=dataset(tmp_path)
    summary=json.loads((report/'summary.json').read_text())
    manifest=json.loads(mani.read_text());manifest['_path']=str(mani)
    manifest['images']=['video-C-frame-000001-foo.png','video-B-frame-000001-bar.png']
    with pytest.raises(ValueError,match='frozen manifest'):
        audit_detection(read_csv(report/'detection_per_image.csv',{'image','method','small_gt'}),summary,manifest)


def test_reject_duplicate_method(tmp_path):
    report,mani,_=dataset(tmp_path)
    summary=json.loads((report/'summary.json').read_text())
    manifest=json.loads(mani.read_text());manifest['_path']=str(mani)
    rows=read_csv(report/'detection_per_image.csv',{'image','method','small_gt'})
    with pytest.raises(ValueError,match='Duplicate'):
        audit_detection(rows+[rows[0]],summary,manifest)


def test_sequence_bootstrap_with_nonzero(tmp_path):
    report,mani,_=dataset(tmp_path)
    summary=json.loads((report/'summary.json').read_text())
    manifest=json.loads(mani.read_text());manifest['_path']=str(mani)
    audited=audit_detection(read_csv(report/'detection_per_image.csv',{'image','method','small_gt'}),summary,manifest)
    changes=paired_deltas(audited,'predictor_bicubic4','random_bicubic4',.25)
    result=sequence_bootstrap_ci(changes,'delta_utility',42,200)
    assert result['mean_per_sequence_equal_weight']==1.75
    assert result['sequence_bootstrap_95ci']==[1.75,1.75]


def test_reject_single_sequence_bootstrap():
    with pytest.raises(ValueError,match='two video'):
        sequence_bootstrap_ci([{'sequence':'a','delta_utility':1}], 'delta_utility', 1, 100)


def test_label_profile(tmp_path):
    _,_,roi=dataset(tmp_path)
    summary=label_profile(read_csv(roi,{'image','tile_index','sequence','delta_fp','delta_small_tp','utility_score','positive_label'}))
    assert summary['counts']['small_gain']==2
    assert summary['counts']['positive_from_fp_only']==2
    assert summary['counts']['positive_utility']==4


def test_label_reject_bad_utility(tmp_path):
    _,_,roi=dataset(tmp_path)
    rows=read_csv(roi,{'image','tile_index','sequence','delta_fp','delta_small_tp','utility_score','positive_label'})
    rows[0]['utility_score']='9'
    with pytest.raises(ValueError,match='Utility formula'):
        label_profile(rows)


def test_full_run_saves_files(tmp_path):
    report,mani,roi=dataset(tmp_path)
    out=tmp_path/'result'
    result=run(report,mani,roi,out,bootstraps=100)
    assert result['images']==2 and result['sequences']==2
    assert len(result['pairwise'])==6
    assert (out/'diagnosis.json').is_file()
    assert (out/'sequence_differences.csv').is_file()
    assert (out/'fp_penalty_sensitivity.csv').is_file()
    assert result['pairwise'][0]['total_utility_difference']==3.5


def test_refuse_overwrite(tmp_path):
    report,mani,roi=dataset(tmp_path)
    out=tmp_path/'existing';out.mkdir()
    with pytest.raises(FileExistsError):run(report,mani,roi,out,bootstraps=100)
