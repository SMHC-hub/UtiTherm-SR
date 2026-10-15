"""Step 11E CPU-only tests including synthetic end-to-end + corruption tests."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_bicubic_first_budget_step11e as budget
import audit_bicubic_first_predictability_step11d as old


def test_parse_config_fixed_protocol(tmp_path):
    src=yaml.safe_load((ROOT/'configs/bicubic_first_budget_step11e.yaml').read_text())
    path=tmp_path/'x.yaml'
    path.write_text(yaml.safe_dump(src))
    assert budget.parse_config(path)['budgets']==[0,1,2,4]
    for field,value in [('budgets',[0,1,3,4]),('scoring','real_detections'),
                        ('policies',['contrast']),('fp_penalty',1),
                        ('contrast_gate_z_thresholds',[.5]),('gate_budgets',[1,2,4]),
                        ('expected_tiles_per_image',4),('bootstrap_iterations',99)]:
        bad=dict(src,**{field:value})
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError):budget.parse_config(path)


@pytest.mark.parametrize('k,expected',[ (0,[]),(1,[2]),(2,[2,0]),(4,[2,0,1,3]) ])
def test_budget_selection(k,expected):
    assert budget.picks(np.array([3.,2.,4.,1.]),k)==expected


def test_normalized_contrast_centered():
    x=np.arange(16,dtype=float)
    z=budget.normalized_contrast(x)
    assert abs(z.mean())<1e-12
    assert abs(z.std()-1)<1e-12


def test_uniform_contrast_can_abstain():
    assert budget.picks(np.ones(16),4,0.0)==[]


def test_gate_threshold_strict():
    x=np.array([1.]*8+[9.]*8)
    assert budget.picks(x,4,0.0)==[8,9,10,11]
    assert budget.picks(x,4,1.0)==[]


def test_nonfinite_rejected():
    with pytest.raises(ValueError):budget.normalized_contrast(np.full(16,np.nan))
    with pytest.raises(ValueError):budget.picks(np.zeros(16),2,float('nan'))


def test_wrong_length_rejected():
    with pytest.raises(ValueError):budget.normalized_contrast(np.ones(15))


def test_measure_counts_and_utility():
    im=['one','two']
    val={'one':np.array([1.,-2.,0.,0.]),'two':np.array([-1.,.5,0.,0.])}
    ranks={'one':np.array([4.,3.,2.,1.]),'two':np.array([1.,4.,3.,2.])}
    res=budget.measure(im,val,ranks,1)
    assert res['selected_regions']==2
    assert res['mean_proxy_per_image']==.75
    assert res['positive_selected']==2
    assert res['negative_selected']==0
    assert res['mean_nominal_processed_lr_pixels_per_image']==8400.


def test_measure_zero_budget():
    im=['a']
    v={'a':np.array([1.,2.,3.])}
    res=budget.measure(im,v,v,0)
    assert res['mean_proxy_per_image']==0
    assert res['images_abstaining']==1
    assert res['selected_regions']==0


def test_random_budget_expectation():
    im=['a','b']; v={'a':np.array([1.,-1.]),'b':np.array([2.,0.])}
    assert budget.random_expected(im,v,1)['mean_proxy_per_image']==.5


def test_gate_val_only_abstains_when_all_harmful():
    ims=['a','b']
    v={im:np.full(16,-1.) for im in ims}
    ranks={im:np.arange(16,dtype=float) for im in ims}
    selected=budget.select_gate(ims,v,ranks,[0,1,2,4],[0.,.5,1.,1.5,2.])
    assert selected['budget']==0
    assert selected['validation_selected_regions']==0


def test_gate_val_only_selects_positive():
    v={'a':np.array([1.]*8+[0.]*8),'b':np.array([1.]*8+[0.]*8)}
    ranks={'a':np.array([9.]*8+[1.]*8),'b':np.array([9.]*8+[1.]*8)}
    selected=budget.select_gate(['a','b'],v,ranks,[0,1,2,4],[0.,.5,1.,1.5,2.])
    assert selected['budget']==4
    assert selected['validation_mean_proxy_per_image']==4


def test_cluster_bootstrap_difference():
    a={'a':1.,'b':1.,'c':1.}; b={'a':0.,'b':0.,'c':0.}; seq={'a':'x','b':'x','c':'y'}
    r=budget.bootstrap_comparison({'per_image':a},{'per_image':b},seq,143,100)
    assert r['sequence_bootstrap_exploratory_95ci']==[1.,1.]
    assert r['image_wins_ties_losses']==[3,0,0]


@pytest.fixture
def synthetic(tmp_path):
    p=tmp_path
    (p/'images').mkdir()
    ids=[f'video-{n}-frame-000001.png' for n in 'abcdefghi']
    utilities={}
    rects=[(c*160,r*128,(c+1)*160,(r+1)*128) for r in range(4) for c in range(4)]
    csvfile=p/'roi.csv'
    with csvfile.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=old.REQUIRED_COLUMNS)
        writer.writeheader()
        for n,im in enumerate(ids):
            rng=np.random.default_rng(100+n)
            gray=rng.integers(0,256,(256,320),dtype=np.uint8)
            cv2.imwrite(str(p/'images'/im),gray)
            utilities[im]=np.zeros(16)
            utilities[im][(n+1)%16]=1.
            utilities[im][(n+2)%16]=.25
            utilities[im][(n+4)%16]=-.25
            for ti,rect in enumerate(rects):
                dtp,dfp=0,0
                if ti==(n+1)%16:dtp=1
                if ti==(n+2)%16:dfp=-1
                if ti==(n+4)%16:dfp=1
                u=dtp-.25*dfp
                writer.writerow(dict(image=im,sequence=old.sequence_id(im),tile_index=ti,
                    tile_x0_hr=rect[0],tile_y0_hr=rect[1],tile_x1_hr=rect[2],tile_y1_hr=rect[3],
                    baseline_kind='full_bicubic_640',processed_lr_pixels=8400,
                    baseline_small_tp=5,baseline_all_tp=7,baseline_fp=5,
                    candidate_small_tp=5+dtp,candidate_all_tp=7+dtp,candidate_fp=5+dfp,
                    delta_small_tp=dtp,delta_all_tp=dtp,delta_fp=dfp,
                    utility_score=u,positive_label=int(u>0)))
    hs=old.sha256(csvfile)
    sr={'protocol':'STEP_11A_INCREMENTAL_LARGE_FSRCNN_OVER_BICUBIC_SINGLE_ROI_V1',
        'baseline':'full_bicubic_640','roi_csv_sha256':hs,'positive_utility':18,
        'source_provenance':{'selected_images':ids}}
    (p/'summary11a.json').write_text(json.dumps(sr))
    seq=[old.sequence_id(im) for im in ids]
    roles={'train':seq[:5],'val':seq[5:7],'test':seq[7:]}
    byroles={role:[im for im in ids if old.sequence_id(im) in seqs] for role,seqs in roles.items()}
    (p/'split.json').write_text(json.dumps({'csv_sha256':hs,'sequence_ids':roles,'images_by_split':byroles}))
    methods={}
    prior_rows=[]
    for im in byroles['test']:
        gray=cv2.imread(str(p/'images'/im),cv2.IMREAD_GRAYSCALE)
        f=old.extract_frame_features(gray,rects)
        u=utilities[im]
        contrast=f[:,old.FEATURES.index('tile_std')]
        edge=f[:,old.FEATURES.index('tile_grad_mean')]
        prior_rows.append({'image':im,'sequence':old.sequence_id(im),
            'contrast_top4':float(sum(u[old.choose_indices(contrast,4)])),
            'edge_top4':float(sum(u[old.choose_indices(edge,4)])),
            'first4':float(sum(u[:4])),
            'random4_expectation':float(4*np.mean(u)),
            'bicubic_no_neural_sr':0.})
    for name in ('contrast','edge'):
        methods[name+'_top4']={'mean_marginal_utility_per_image':float(np.mean([r[name+'_top4'] for r in prior_rows]))}
    prior={'protocol':'STEP_11D_BICUBIC_FIRST_CHEAP_FEATURE_PREDICTABILITY_V1',
        'source_csv_sha256':hs,'train_val_test_image_counts':{k:len(v) for k,v in byroles.items()},
        'train_val_test_sequence_counts':{k:len(v) for k,v in roles.items()},
        'results':{'test':{'methods':methods}}}
    (p/'summary11d.json').write_text(json.dumps(prior))
    with (p/'perimage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=prior_rows[0].keys());w.writeheader();w.writerows(prior_rows)
    cfg=yaml.safe_load((ROOT/'configs/bicubic_first_budget_step11e.yaml').read_text())
    cfg.update(source_csv=str(csvfile),source_summary=str(p/'summary11a.json'),
        step11d_split=str(p/'split.json'),step11d_summary=str(p/'summary11d.json'),
        step11d_test_per_image=str(p/'perimage.csv'),train_lr=str(p/'images'),
        out=str(p/'output'),expected_images=9,expected_sequences=9,expected_interventions=144,
        bootstrap_iterations=100)
    (p/'config.yaml').write_text(yaml.safe_dump(cfg))
    return p


def test_source_valid(synthetic):
    cfg=budget.parse_config(synthetic/'config.yaml')
    rows,group,split,seq,prior,perimage,h=budget.verify_sources(cfg)
    assert len(rows)==144 and len(group)==9
    assert len(split['test'])==2


def test_replay_alignment_valid(synthetic):
    cfg=budget.parse_config(synthetic/'config.yaml')
    rows,group,sets,seq,prior,perimage,_=budget.verify_sources(cfg)
    u,c,e={},{},{}
    for im in sets['test']:
        gray=cv2.imread(str(synthetic/'images'/im),0)
        rr=sorted(group[im],key=lambda r:r['tile_int'])
        rects=[tuple(int(r[k]) for k in ('tile_x0_hr','tile_y0_hr','tile_x1_hr','tile_y1_hr')) for r in rr]
        f=old.extract_frame_features(gray,rects)
        u[im]=np.array([r['utility_float'] for r in rr]);c[im]=f[:,old.FEATURES.index('tile_std')]
        e[im]=f[:,old.FEATURES.index('tile_grad_mean')]
    budget.prior_step11d_test_alignment(sets['test'],u,c,e,prior,perimage)


def test_source_hash_corruption_fails(synthetic):
    csvpath=synthetic/'roi.csv'
    csvpath.write_text(csvpath.read_text()+'\n')
    with pytest.raises(ValueError,match='hash|protocol'):budget.verify_sources(budget.parse_config(synthetic/'config.yaml'))


def test_overlap_fails(synthetic):
    path=synthetic/'split.json'; d=json.loads(path.read_text())
    d['sequence_ids']['val'].append(d['sequence_ids']['test'][0])
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='overlap'):budget.verify_sources(budget.parse_config(synthetic/'config.yaml'))


def test_test_replay_corruption_fails(synthetic):
    path=synthetic/'perimage.csv'
    lines=path.read_text().splitlines()
    lines[1]=lines[1].replace('video-', 'other-',1)
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='align'):budget.run(Namespace(config=synthetic/'config.yaml',out=None,overwrite=False))


def test_synthetic_end_to_end(synthetic):
    result=budget.run(Namespace(config=synthetic/'config.yaml',out=None,overwrite=False))
    assert result['source_interventions']==144
    assert result['results']['test']['num_images']==2
    assert result['validation_selected_contrast_gate']['num_candidates']==16
    assert (synthetic/'output'/'summary.json').exists()
    with (synthetic/'output'/'per_image_budget_proxy.csv').open(newline='') as f:
        assert len(list(csv.DictReader(f)))==4
    with pytest.raises(FileExistsError):budget.run(Namespace(config=synthetic/'config.yaml',out=None,overwrite=False))


def test_synthetic_output_does_not_override_source(synthetic):
    before=old.sha256(synthetic/'roi.csv')
    budget.run(Namespace(config=synthetic/'config.yaml',out=None,overwrite=False))
    assert before==old.sha256(synthetic/'roi.csv')


def test_wrong_prior_summary_fails(synthetic):
    path=synthetic/'summary11d.json'; d=json.loads(path.read_text())
    d['source_csv_sha256']='wrong';path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='hash'):budget.verify_sources(budget.parse_config(synthetic/'config.yaml'))
