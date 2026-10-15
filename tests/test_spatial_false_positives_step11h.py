"""CPU tests for Step11H spatial and matching invariants (no CUDA needed)."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import json
import sys
import math
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_spatial_false_positives_step11h import (match_fp_indices,pair_false_positives,
                                                   locate_fp,record_map,PROTOCOL)

def iou(a,b):
    l=max(a[0],b[0]);r=min(a[2],b[2]);t=max(a[1],b[1]);bb=min(a[3],b[3]);
    inter=max(0,r-l)*max(0,bb-t)
    total=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/total if total else 0.

def p(box,cls=0,conf=.9):
    return {'box':box,'cls':cls,'conf':conf}

TILES=[(c*160,r*128,(c+1)*160,(r+1)*128) for r in range(4) for c in range(4)]

@pytest.mark.parametrize('rect,expected,overlap',[
    ((10,10,20,20),'selected_interior',1),
    ((1,20,3,24),'selected_boundary_band',1),
    ((90,10,110,20),'outside_selected_tiles',0),
    ((70,10,85,20),'selected_boundary_band',1),
    ((77,60,81,64),'selected_boundary_band',1),
    ((72,10,92,20),'outside_selected_tiles',1),
])
def test_locate(rect,expected,overlap):
    ans=locate_fp(rect,[0],TILES,8)
    assert ans['centroid_zone']==expected
    assert ans['bbox_overlaps_selected']==overlap

def test_tile_one_center():
    ans=locate_fp((85,10,90,20),[1,10,12,15],TILES,8)
    assert ans['centroid_zone']=='selected_interior'
    assert ans['center_tile']==1

def test_overlap_fraction_exact():
    ans=locate_fp((70,20,90,40),[0],TILES,8)
    assert ans['box_selected_overlap_fraction']==.5

def test_center_boundary_eight_exact():
    # centre is precisely 8 HR pixels from left tile edge: interior.
    ans=locate_fp((3,10,5,12),[0],TILES,8)
    assert ans['centroid_zone']=='selected_interior'

@pytest.mark.parametrize('selected,boundary,rect',[
    ([],8,(1,1,2,2)),([0,0],8,(1,1,2,2)),([17],8,(1,1,2,2)),
    ([0],-1,(1,1,2,2)),([0],33,(1,1,2,2)),
    ([0],8,(-1,1,2,2)),([0],8,(1,1,330,2)),
    ([0],8,(2,1,1,2)),([0],8,(1,1,float('nan'),2)),
])
def test_locate_invalid(selected,boundary,rect):
    with pytest.raises(ValueError):locate_fp(rect,selected,TILES,boundary)

def test_gt_match_class_aware():
    gt=[{'cls':1,'box':(0,0,10,10),'size':'small'}]
    preds=[p((0,0,10,10),cls=0),p((0,0,10,10),cls=1),p((0,0,10,10),cls=1)]
    assert match_fp_indices(gt,preds,.25,iou)==[0,2]

def test_gt_match_conf_cutoff():
    assert match_fp_indices([], [p((0,0,1,1),conf=.249)],.25,iou)==[]

def test_gt_match_iou_half():
    gt=[{'cls':0,'box':(0,0,10,10),'size':'small'}]
    assert match_fp_indices(gt,[p((0,0,10,10))],.25,iou)==[]
    assert match_fp_indices(gt,[p((20,20,30,30))],.25,iou)==[0]

def test_gt_match_fp_order():
    assert match_fp_indices([], [p((0,0,1,1),conf=.1),p((0,0,1,1),conf=.8)],.25,iou)==[1]

def test_pair_one_same():
    before=[p((0,0,10,10))];after=[p((1,0,11,10))]
    pairs,a,r=pair_false_positives(before,after,[0],[0],.5,iou)
    assert len(pairs)==1 and not a and not r

def test_pair_add_one():
    before=[p((0,0,10,10))]
    after=[p((0,0,10,10)),p((50,50,60,60))]
    pairs,a,r=pair_false_positives(before,after,[0],[0,1],.5,iou)
    assert a==[1] and r==[]

def test_pair_remove():
    before=[p((0,0,10,10)),p((50,50,60,60))]
    after=[p((0,0,10,10))]
    pairs,a,r=pair_false_positives(before,after,[0,1],[0],.5,iou)
    assert a==[] and r==[1]

def test_pair_class_different():
    before=[p((0,0,10,10),cls=0)];after=[p((0,0,10,10),cls=1)]
    pairs,a,r=pair_false_positives(before,after,[0],[0],.5,iou)
    assert not pairs and a==[0] and r==[0]

def test_pair_one_to_one():
    before=[p((0,0,10,10))]
    after=[p((0,0,10,10)),p((0,0,10,10))]
    pairs,a,r=pair_false_positives(before,after,[0],[0,1],.5,iou)
    assert pairs[0][:2]==(0,0) and a==[1] and not r

def test_pair_duplicate_rejected():
    with pytest.raises(ValueError):pair_false_positives([p((0,0,1,1))],[],[0,0],[],.5,iou)

@pytest.mark.parametrize('pair_iou',[0,-.1,1.01])
def test_pair_bad_threshold(pair_iou):
    with pytest.raises(ValueError):pair_false_positives([],[],[],[],pair_iou,iou)

def test_reference_duplicate_rejected():
    with pytest.raises(ValueError):record_map([{'image':'a','method':'z'},{'image':'a','method':'z'}])

def test_reference_key():
    assert record_map([{'image':'a','method':'z'}])[('a','z')]['image']=='a'

def test_protocol_fixed():
    assert PROTOCOL.endswith('_V1')
