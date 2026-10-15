"""Data-leakage and reporting safeguards for Step 7A; no GPU or FLIR needed."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from prepare_development_split import (
    sequence_id, sha256, training_sequences, eligible_paths, select_diverse,
    check_manifest,
)
from evaluate_development_holdout import MODES, assigned_tiles, analysis


def touch(path:Path, text=''):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')


@pytest.fixture
def tiny(tmp_path):
    train=tmp_path/'images'/'train'
    val=tmp_path/'images'/'val'
    labels=tmp_path/'labels'/'val'
    train_files=['video-trA-frame-000001-a.png','video-trB-frame-000002-a.png','video-trC-frame-000001-a.png']
    for name in train_files: touch(train/name)
    for seq in ('video-vaA','video-vaB','video-vaC','video-trA'):
        for i in range(3):
            name=f'{seq}-frame-{i:06d}-a.png'
            touch(val/name); touch(labels/(Path(name).stem+'.txt'))
    csv_path=tmp_path/'measurements.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['sequence','image'])
        w.writeheader()
        for name in train_files: w.writerow({'sequence':sequence_id(name),'image':name})
    manifest=tmp_path/'train_split.json'
    touch(manifest,json.dumps({'train_sequences':['video-trA','video-trB'],
                               'val_sequences':['video-trC'],
                               'intervention_csv_sha256':sha256(csv_path)}))
    return train,val,labels,csv_path,manifest


def test_sequence_parser():
    assert sequence_id('video-abc-frame-001230-x.png')=='video-abc'
    with pytest.raises(ValueError): sequence_id('../file.png')
    with pytest.raises(ValueError): sequence_id('anything.png')
    with pytest.raises(ValueError): sequence_id('video-abc-frame-111.jpg')


def test_training_manifest_source_integrity(tiny):
    tr,va,la,csv_path,split=tiny
    assert training_sequences(tr,csv_path,split)=={'video-trA','video-trB','video-trC'}


def test_reject_changed_label_source(tiny):
    tr,va,la,csv_path,split=tiny
    csv_path.write_text(csv_path.read_text()+'altered,data\n')
    with pytest.raises(ValueError,match='disagree'):
        training_sequences(tr,csv_path,split)


def test_excludes_detector_train_sequences(tiny):
    tr,va,la,me,sp=tiny
    eligible,stats=eligible_paths(va,la,training_sequences(tr,me,sp))
    assert len(eligible)==9
    assert stats['sequence_collisions_excluded']==3
    assert stats['eligible_unique_sequences']==3


def test_missing_labels_fail_loudly(tiny):
    tr,va,la,me,sp=tiny
    next(la.glob('video-vaA*.txt')).unlink()
    with pytest.raises(ValueError,match='lack label'):
        eligible_paths(va,la,training_sequences(tr,me,sp))


def test_sampler_diverse_deterministic(tiny):
    tr,va,la,me,sp=tiny
    eligible,_=eligible_paths(va,la,training_sequences(tr,me,sp))
    a=select_diverse(eligible,6,seed=42,max_per_sequence=2)
    b=select_diverse(eligible,6,seed=42,max_per_sequence=2)
    assert a==b
    assert len({sequence_id(p.name) for p in a[:3]})==3
    assert {sequence_id(p.name) for p in a}=={'video-vaA','video-vaB','video-vaC'}


def test_sampler_insufficient_limit(tiny):
    tr,va,la,me,sp=tiny
    eligible,_=eligible_paths(va,la,training_sequences(tr,me,sp))
    assert len(select_diverse(eligible,20,42,2))==6


def test_sampler_validation():
    with pytest.raises(ValueError): select_diverse([],0,42,2)
    with pytest.raises(ValueError): select_diverse([],1,42,0)


def manifest_for(tiny):
    tr,va,la,me,sp=tiny
    eligible,_=eligible_paths(va,la,training_sequences(tr,me,sp))
    names=[p.name for p in select_diverse(eligible,6,42,2)]
    return {'protocol':'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST',
            'utility_csv_sha256':sha256(me),'predictor_split_sha256':sha256(sp),
            'images':names,'selected_images':6,'unique_sequences':3,'max_per_sequence':2,
            'eval_lr_directory':str(va.resolve()),
            'image_sha256':{n:sha256(va/n) for n in names},
            'label_sha256':{n:sha256(la/(Path(n).stem+'.txt')) for n in names}}


def test_frozen_manifest_valid(tiny):
    tr,va,la,me,sp=tiny
    m=manifest_for(tiny)
    assert check_manifest(m,va,la,training_sequences(tr,me,sp),me,sp)==m['images']


def test_frozen_manifest_reject_train_overlap(tiny):
    tr,va,la,me,sp=tiny
    m=manifest_for(tiny)
    m['images'][0]='video-trA-frame-000000-a.png'
    with pytest.raises(ValueError,match='share a sequence'):
        check_manifest(m,va,la,training_sequences(tr,me,sp),me,sp)


def test_frozen_manifest_reject_duplicates(tiny):
    tr,va,la,me,sp=tiny
    m=manifest_for(tiny)
    m['images'][1]=m['images'][0]
    with pytest.raises(ValueError,match='duplicate'):
        check_manifest(m,va,la,training_sequences(tr,me,sp),me,sp)


def test_frozen_manifest_reject_changed_source(tiny):
    tr,va,la,me,sp=tiny
    m=manifest_for(tiny)
    m['predictor_split_sha256']='0000'
    with pytest.raises(ValueError,match='changed'):
        check_manifest(m,va,la,training_sequences(tr,me,sp),me,sp)


def test_methods_have_controls_and_baselines():
    assert len(MODES)==8
    assert 'random_bicubic4' in MODES
    assert 'predictor_bicubic4' in MODES
    assert 'predictor_batched_fsrcnn4' in MODES
    assert 'full_bicubic_640' in MODES


def test_selector_is_deterministic():
    import numpy as np
    scores=np.arange(16,dtype=float)
    fixed=assigned_tiles('fixed_bicubic4','video-z-frame-01.png',scores,42)
    random1=assigned_tiles('random_bicubic4','video-z-frame-01.png',scores,42)
    random2=assigned_tiles('random_bicubic4','video-z-frame-01.png',scores,42)
    predicted=assigned_tiles('predictor_bicubic4','video-z-frame-01.png',scores,42)
    assert fixed==[0,1,2,3] and random1==random2
    assert predicted==[15,14,13,12]


def test_cluster_based_metrics_and_complete_trial_enforcement():
    image_names=['video-s1-frame-0001.png','video-s1-frame-0002.png',
                 'video-s2-frame-0001.png','video-s3-frame-0001.png']
    trials=[]; det=[]
    for im in image_names:
        for rep in range(3):
            for mode in MODES:
                trials.append({'image':im,'method':mode,'repeat':rep,
                               'total_ms':10+MODE_INDEX[mode]+rep*.1,'selector_ms':.5,
                               'sr_ms':1.})
        for mode in MODES:
            det.append({'image':im,'method':mode,'small_gt':10,
                        'small_tp':5 + (mode=='predictor_bicubic4'),
                        'all_tp':6 + (mode=='predictor_bicubic4'),
                        'fp':2})
    out=analysis(trials,det,42,n_boot=100)
    assert out['unique_sequences']==3 and out['images']==4
    assert out['methods']['predictor_bicubic4']['small_tp']==24
    assert out['paired_comparisons']['predictor_bicubic4']['utility_total_vs_linear']==4
    with pytest.raises(ValueError,match='Missing trials'):
        analysis(trials[:-1],det,42,n_boot=100)
    with pytest.raises(ValueError,match='Duplicate trial'):
        analysis(trials+[trials[0]],det,42,n_boot=100)


MODE_INDEX={m:i for i,m in enumerate(MODES)}


def test_frozen_manifest_reject_changed_image(tiny):
    tr,va,la,me,sp=tiny
    m=manifest_for(tiny)
    touch(va/m['images'][0],'modified pixels')
    with pytest.raises(ValueError,match='contents changed'):
        check_manifest(m,va,la,training_sequences(tr,me,sp),me,sp)


def test_frozen_manifest_reject_changed_annotation(tiny):
    tr,va,la,me,sp=tiny
    m=manifest_for(tiny)
    touch(la/(Path(m['images'][0]).stem+'.txt'),'0 0.5 0.5 0.1 0.1')
    with pytest.raises(ValueError,match='contents changed'):
        check_manifest(m,va,la,training_sequences(tr,me,sp),me,sp)
