"""CPU-only tests for Step 11D. No detector, GPU, or FLIR data required."""
from __future__ import annotations

import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_bicubic_first_predictability_step11d as audit


def synthetic_rgb(seed=143):
    rng=np.random.default_rng(seed)
    return rng.integers(0,256,(256,320),dtype=np.uint8)


def rectangles():
    return [(c*160,r*128,(c+1)*160,(r+1)*128) for r in range(4) for c in range(4)]


def make_row(image,tile,du=0.0,basefp=5):
    x0,y0,x1,y1=rectangles()[tile]
    dtp=int(du) if float(du).is_integer() else 0
    dfp=int(round((dtp-du)*4))
    return dict(image=image,sequence=audit.sequence_id(image),tile_index=tile,
        tile_x0_hr=x0,tile_y0_hr=y0,tile_x1_hr=x1,tile_y1_hr=y1,
        baseline_kind='full_bicubic_640',processed_lr_pixels=8400,
        baseline_small_tp=5,baseline_all_tp=7,baseline_fp=basefp,
        candidate_small_tp=5+dtp,candidate_all_tp=7+dtp,candidate_fp=basefp+dfp,
        delta_small_tp=dtp,delta_all_tp=dtp,delta_fp=dfp,
        utility_score=du,positive_label=int(du>0))


@pytest.fixture
def onecsv(tmp_path):
    img='video-a-frame-000001.png'
    path=tmp_path/'roi.csv'
    with path.open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=audit.REQUIRED_COLUMNS)
        writer.writeheader()
        for i in range(16):writer.writerow(make_row(img,i,du=1 if i==4 else 0))
    return path


@pytest.mark.parametrize('seed',[0,1,3,14,79,99,110,143])
def test_feature_finite_and_shape(seed):
    features=audit.extract_frame_features(synthetic_rgb(seed),rectangles())
    assert features.shape==(16,len(audit.FEATURES))
    assert np.isfinite(features).all()


def test_features_do_not_require_labels():
    f=audit.extract_frame_features(np.full((256,320),120,np.uint8),rectangles())
    assert np.allclose(f[:,audit.FEATURES.index('tile_mean')],120)
    assert np.allclose(f[:,audit.FEATURES.index('tile_std')],0)
    assert np.isfinite(f).all()


@pytest.mark.parametrize('bad_shape',[(16,16),(512,640),(256,319)])
def test_shape_fails(bad_shape):
    with pytest.raises(ValueError,match='shape'):
        audit.extract_frame_features(np.zeros(bad_shape,np.uint8),rectangles())


def test_rgb_fails():
    with pytest.raises(ValueError,match='grayscale'):
        audit.extract_frame_features(np.zeros((256,320,3),np.uint8),rectangles())


def test_invalid_tile_coords():
    with pytest.raises(ValueError,match='rectangle'):
        audit.extract_frame_features(synthetic_rgb(),[(0,0,161,128)])


def test_csv_valid(onecsv):
    rows,gr=audit.validated_rows(onecsv,16,.25)
    assert len(rows)==16
    assert len(gr)==1
    assert sum(x['positive_int'] for x in rows)==1


@pytest.mark.parametrize('field,value',[
    ('baseline_kind','linear_640'),('processed_lr_pixels','3000'),
    ('delta_fp','999'),('positive_label','0'),('tile_x0_hr','9'),
    ('sequence','video-other'),('utility_score','999'),
])
def test_csv_corruption_fails(onecsv,field,value):
    with onecsv.open(newline='') as f:
        r=list(csv.DictReader(f))
    r[4][field]=value
    with onecsv.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=audit.REQUIRED_COLUMNS)
        w.writeheader(); w.writerows(r)
    with pytest.raises(ValueError):audit.validated_rows(onecsv,16,.25)


def test_duplicate_tile_fails(onecsv):
    with onecsv.open(newline='') as f:r=list(csv.DictReader(f))
    r[15]=dict(r[0])
    with onecsv.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=audit.REQUIRED_COLUMNS)
        w.writeheader();w.writerows(r)
    with pytest.raises(ValueError,match='Duplicate'):audit.validated_rows(onecsv,16,.25)


def test_missing_tile_fails(onecsv):
    lines=onecsv.read_text().splitlines()
    onecsv.write_text('\n'.join(lines[:-1])+'\n')
    with pytest.raises(ValueError,match='Incomplete'):audit.validated_rows(onecsv,16,.25)


@pytest.mark.parametrize('seed',[1,143,2026])
def test_split_is_disjoint_and_deterministic(seed):
    seq=[f's{i}' for i in range(33)]
    a=audit.split_groups(seq,seed,.15,.15)
    b=audit.split_groups(list(reversed(seq)),seed,.15,.15)
    assert a==b
    assert [len(a[x]) for x in ('train','val','test')]==[23,5,5]
    assert len(set(a['train'])|set(a['val'])|set(a['test']))==33
    assert not (set(a['train'])&set(a['val']) or set(a['train'])&set(a['test']) or set(a['val'])&set(a['test']))


def test_split_rejects_invalid_fraction():
    with pytest.raises(ValueError):audit.split_groups(list('abcdef'),143,.6,.15)


def test_split_rejects_few_sequences():
    with pytest.raises(ValueError):audit.split_groups(list('abcde'),143,.15,.15)


def test_tied_scores_choose_lowest_index():
    assert audit.choose_indices(np.ones(16),4)==[0,1,2,3]


def test_ranking_and_budget():
    assert audit.choose_indices(np.array([.5,.1,.9,-1]),2)==[2,0]


def test_gating_can_abstain():
    assert audit.choose_indices(np.array([0.,-.1,-.3]),2,0.0)==[]


def test_gating_strict_threshold():
    assert audit.choose_indices(np.array([.25,.26,.3]),3,.25)==[2,1]


def test_ranking_invalid_nan():
    with pytest.raises(ValueError):audit.choose_indices(np.array([np.nan,1]),1)


def test_budget_limit():
    with pytest.raises(ValueError):audit.choose_indices(np.array([1.,2.]),3)


def test_random_expectation_is_analytic():
    v={'img':np.arange(16,dtype=float)}
    out=audit.exact_random_expectation(['img'],v,4)
    assert out['mean_marginal_utility_per_image']==30


def test_oracle_top4_and_gated_differ():
    v=np.array([2.,-1.,-2.,0.,-5.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    assert sum(v[audit.choose_indices(v,4)])==2.0
    assert audit.choose_indices(v,4,0.)==[0]


def test_bootstrap_constant_group_diff():
    a={'a':2.,'b':2.,'c':2.}
    b={'a':1.,'b':1.,'c':1.}
    seq={'a':'x','b':'x','c':'y'}
    assert audit.cluster_bootstrap_difference(a,b,seq,123,500)==(1.,1.)


def test_bootstrap_rejects_unmatched_keys():
    with pytest.raises(ValueError):
        audit.cluster_bootstrap_difference({'a':1},{'b':2},{'a':'x'},143,500)


def test_safe_ap_requires_both_classes():
    assert audit.safe_ap(np.array([0,0]),np.array([.3,.4])) is None
    assert audit.safe_ap(np.array([0,1]),np.array([.2,.8]))==1.


def test_score_counts_positive_negative_and_abstention():
    vals={'i':np.array([.25,-.25,0.,.5])}
    ranks={'i':np.array([4.,3.,2.,1.])}
    out=audit.score_selections(['i'],vals,ranks,2)
    assert out['mean_marginal_utility_per_image']==0
    assert out['selected_regions']==2
    assert out['positive_selected']==1 and out['negative_selected']==1
    none=audit.score_selections(['i'],vals,ranks,2,threshold=5)
    assert none['images_abstaining']==1


@pytest.fixture
def synthetic_project(tmp_path):
    """Small end-to-end fixture: no FLIR files, no detector/GPU."""
    directory=tmp_path/'images';directory.mkdir()
    rows=[];imgs=[]
    for seq in range(8):
        for j in range(2):
            image=f'video-test{seq:02d}-frame-{j:06d}.png'
            frame=np.full((256,320),30,dtype=np.uint8)
            for t in range(16):
                col,rr=t%4,t//4
                v=(18*t+7*seq+11*j)%230
                frame[64*rr:64*(rr+1),80*col:80*(col+1)]=v
                # utility from LR pixels; includes both positive/negative.
                delta=1.0 if v>180 else (-.25 if v<50 else 0.0)
                rows.append(make_row(image,t,delta,basefp=10))
            cv2.imwrite(str(directory/image),frame)
            imgs.append(image)
    csv_path=tmp_path/'incremental_roi_utility.csv'
    with csv_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=audit.REQUIRED_COLUMNS);w.writeheader();w.writerows(rows)
    source={'protocol':'STEP_11A_INCREMENTAL_LARGE_FSRCNN_OVER_BICUBIC_SINGLE_ROI_V1',
            'roi_csv_sha256':audit.sha256(csv_path),'baseline':'full_bicubic_640',
            'positive_utility':sum(float(r['utility_score'])>0 for r in rows),
            'source_provenance':{'selected_images':imgs}}
    sourcefile=tmp_path/'summary.json';sourcefile.write_text(json.dumps(source))
    config={'source_csv':str(csv_path),'source_summary':str(sourcefile),'train_lr':str(directory),
            'out':str(tmp_path/'out'),'expected_images':len(imgs),'expected_sequences':8,
            'expected_tiles_per_image':16,'expected_interventions':len(rows),'fp_penalty':.25,
            'seed':143,'val_sequence_fraction':.15,'test_sequence_fraction':.15,'budget_tiles':4,
            'bootstrap_iterations':150,'image_width_lr':320,'image_height_lr':256,
            'scoring':'marginal_one_tile_utility_sum_proxy_ONLY'}
    cfgfile=tmp_path/'cfg.yaml';cfgfile.write_text(yaml.safe_dump(config))
    return cfgfile,tmp_path/'out',csv_path


def test_end_to_end_synthetic_run(synthetic_project):
    cfg,out,source=synthetic_project
    summary=audit.run(Namespace(config=cfg,out=None,overwrite=False))
    assert summary['num_tiles']==256
    assert summary['num_images']==16
    assert summary['num_sequences']==8
    assert (out/'summary.json').is_file()
    assert (out/'test_per_image.csv').is_file()
    assert (out/'sequence_split.json').is_file()
    assert 'extra_trees_gated' in summary['results']['test']['methods']
    assert 'extra_trees_top4__vs__random4_expectation' in summary['results']['test']['comparisons']
    assert summary['results']['test']['methods']['bicubic_no_neural_sr']['mean_marginal_utility_per_image']==0
    with pytest.raises(FileExistsError): audit.run(Namespace(config=cfg,out=None,overwrite=False))


def test_refuses_checksum_mismatch(synthetic_project):
    cfg,out,source=synthetic_project
    source.write_text(source.read_text()+'\n')
    # checksum mismatch due to changed source CSV bytes but unchanged source summary.
    with source.open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='Source summary'):
        audit.run(Namespace(config=cfg,out=None,overwrite=False))
