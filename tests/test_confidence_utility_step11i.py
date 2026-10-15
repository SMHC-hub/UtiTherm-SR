"""Pure CPU unit tests, no FLIR or GPU required."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_confidence_utility_step11i import (
    BASE,CONTRAST,THRESHOLDS,PROTOCOL,match_detail,compare_against_official,
    delta_records,validate_base_row,validate_thresholds,summarize,
)


def iou(a,b):
    x0=max(a[0],b[0]);x1=min(a[2],b[2]);y0=max(a[1],b[1]);y1=min(a[3],b[3]);
    intersection=max(0,x1-x0)*max(0,y1-y0)
    area=lambda r:(r[2]-r[0])*(r[3]-r[1])
    return intersection/(area(a)+area(b)-intersection) if area(a)+area(b)>intersection else 0


def pred(box=(0,0,10,10), cls=0, conf=.7):
    return {'box':box,'cls':cls,'conf':conf}


def gt(box=(0,0,10,10), cls=0, size='small'):
    return {'box':box,'cls':cls,'size':size}


@pytest.mark.parametrize('conf,expected',[(.25,(1,2)),(.30,(1,1)),(.35,(1,0)),(.5,(1,0)),(.8,(0,0))])
def test_confidence_cutoff(conf,expected):
    g=[gt()]
    p=[pred(conf=.7),pred((15,15,30,30),conf=.28),pred((100,100,110,110),conf=.31)]
    d=match_detail(g,p,conf,iou)
    assert (d['small_tp'],d['fp'])==expected


def test_exact_threshold_inclusive():
    assert match_detail([], [pred(conf=.35)], .35, iou)['fp']==1


def test_wrong_class_is_fp():
    d=match_detail([gt(cls=1)],[pred(cls=0)],.25,iou)
    assert d['small_tp']==0 and d['fp']==1


def test_medium_gt_not_small():
    d=match_detail([gt(size='medium')],[pred()],.25,iou)
    assert d['small_tp']==0 and d['all_tp']==1 and d['small_gt']==0


def test_greedy_matching_in_confidence_order():
    g=[gt(),gt((20,20,30,30))]
    p=[pred(conf=.4),pred((20,20,30,30),conf=.8)]
    d=match_detail(g,p,.25,iou)
    assert d['matched']=={1:1,0:0} and d['small_gt_indices']=={0,1}


def test_gt_tie_latest_iou():
    d=match_detail([gt(),gt()], [pred()],.25,iou)
    assert d['matched']=={1:0} # official uses >=, last GT wins tie


def test_gt_one_to_one_duplicate_prediction():
    d=match_detail([gt()], [pred(conf=.9),pred(conf=.8)],.25,iou)
    assert d['small_tp']==1 and d['fp']==1 and d['matched']=={0:0}


def test_detailed_confidence_matching_identical_to_official_mock():
    g=[gt(),gt(cls=1)]
    p=[pred(conf=.8),pred(cls=1,conf=.7),pred(conf=.6),pred(cls=4,conf=.3)]
    d=match_detail(g,p,.25,iou)
    def counts(a,b,threshold):
        z=match_detail(a,b,threshold,iou)
        return {k:z[k] for k in ('small_gt','small_tp','all_tp','fp')}
    def matches(a,b,threshold):
        z=match_detail(a,b,threshold,iou)
        return set(z['matched']),z['fp']
    compare_against_official(d,g,p,.25,counts,matches)


def test_official_mismatch_fails():
    d=match_detail([],[],.25,iou)
    def wrong(*a): return {'small_gt':0,'small_tp':0,'all_tp':0,'fp':1}
    with pytest.raises(RuntimeError,match='Matcher mismatch'):
        compare_against_official(d,[],[],.25,wrong,lambda *a:(set(),0))


def test_official_ids_mismatch_fails():
    d=match_detail([gt()],[pred()],.25,iou)
    def good(*a): return {k:d[k] for k in ('small_gt','small_tp','all_tp','fp')}
    with pytest.raises(RuntimeError,match='GT identities'):
        compare_against_official(d,[gt()],[pred()],.25,good,lambda *a:(set(),0))


@pytest.mark.parametrize('invalid',[[.25,.30],[.25,.35,.30,.4,.5],[.25,.30,.35,.4,.5,.6],
                                    [0,.3,.35,.4,.5],[float('nan'),.30,.35,.40,.5],
                                    [.25,.30,.35,.40,.50,.50]])
def test_thresholds_locked(invalid):
    with pytest.raises(ValueError): validate_thresholds(invalid)


def test_thresholds_valid():
    assert validate_thresholds(list(THRESHOLDS))==THRESHOLDS


@pytest.mark.parametrize('confidence',[-.1,1.1,float('nan')])
def test_bad_confidence(confidence):
    with pytest.raises(ValueError): match_detail([],[],confidence,iou)


def test_tp_gained_lost_identity():
    base={'small_gt':5,'small_gt_indices':{0,1,2},'small_tp':3,'fp':4}
    contrast={'small_gt':5,'small_gt_indices':{1,2,3,4},'small_tp':4,'fp':7}
    d=delta_records(base,contrast,.25)
    assert d=={'gained_small_gt':2,'lost_small_gt':1,
               'delta_small_tp':1,'delta_fp':3,'delta_utility':.25}


def test_tp_wrong_gt_count():
    with pytest.raises(ValueError): delta_records(
        {'small_gt':2,'small_gt_indices':set(),'small_tp':0,'fp':0},
        {'small_gt':3,'small_gt_indices':set(),'small_tp':0,'fp':0},.25)


def test_same_tp_count_can_hide_recovery_churn():
    d=delta_records({'small_gt':2,'small_gt_indices':{0},'small_tp':1,'fp':0},
                    {'small_gt':2,'small_gt_indices':{1},'small_tp':1,'fp':0},.25)
    assert d['delta_small_tp']==0 and d['gained_small_gt']==1 and d['lost_small_gt']==1


def test_base_row_exact_reference():
    cur={'small_gt':2,'small_tp':1,'all_tp':1,'fp':3}
    ref={('file','method'):dict(cur)}
    validate_base_row(cur,ref,('file','method'),.25)
    validate_base_row({'fp':999},ref,('file','method'),.3)
    with pytest.raises(RuntimeError):validate_base_row({**cur,'fp':4},ref,('file','method'),.25)


def test_summary_two_sequences_and_conservation():
    rows=[]
    records=[('a','seq1',{0},1,{0,1},2),('b','seq2',{0},2,{0},1)]
    for im,seq,base_g,base_fp,contrast_g,contrast_fp in records:
        for m,ids,fp_count in ((BASE,base_g,base_fp),(CONTRAST,contrast_g,contrast_fp)):
            rows.append({'image':im,'sequence':seq,'method':m,'threshold':'.25',
                'small_gt':3,'small_tp':len(ids),'all_tp':len(ids),'fp':fp_count,
                'matched_small_gt_indices':__import__('json').dumps(sorted(ids))})
    results=summarize(rows,(.25,), .25,211,100)
    r=results['0.25']
    assert r['methods'][BASE]['small_tp']==2
    assert r['methods'][CONTRAST]['small_tp']==3
    assert r['contrast_vs_bicubic']['delta_small_tp']==1
    assert r['contrast_vs_bicubic']['delta_fp']==0
    assert r['contrast_vs_bicubic']['delta_utility_total']==1
    assert r['contrast_vs_bicubic']['gained_small_gt']==1


def test_summary_missing_method_rejected():
    rows=[{'image':'a','sequence':'s1','method':BASE,'threshold':'.25',
            'small_gt':1,'small_tp':0,'all_tp':0,'fp':0,'matched_small_gt_indices':'[]'},
          {'image':'b','sequence':'s2','method':BASE,'threshold':'.25',
            'small_gt':1,'small_tp':0,'all_tp':0,'fp':0,'matched_small_gt_indices':'[]'}]
    with pytest.raises(ValueError,match='Missing'):
        summarize(rows,(.25,),.25,211,100)


def test_summary_duplicate_rejected():
    row={'image':'a','sequence':'s1','method':BASE,'threshold':'.25'}
    with pytest.raises(ValueError,match='Duplicate'):
        summarize([row,row],(.25,),.25,211,100)


def test_protocol_version():
    assert PROTOCOL.endswith('_V1')
