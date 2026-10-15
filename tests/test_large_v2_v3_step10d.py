"""Offline protocol and analysis tests; run in the existing research container."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_v2_v3_large_step10d as step10d


def pconfig():
    exp = {'halos':[10,16],'budget_tiles':4,'false_positive_penalty':.25,
           'expected_images':200,'expected_sequences':17}
    base = {'imgsz':640,'budget_tiles':4,'feather_hr_px':8,'confidence':.25,
            'nms_iou':.7,'false_positive_penalty':.25,'device':'0'}
    manifest = {'selected_images':200,'unique_sequences':17}
    return exp,base,manifest


def test_nine_unique_modes():
    assert len(step10d.MODES)==len(set(step10d.MODES))==9


def test_essential_controls():
    assert {'linear_640','full_bicubic_640','full_large_fsrcnn_640'} <= set(step10d.MODES)


@pytest.mark.parametrize('h',[10,16])
def test_paired_roi_methods_for_both_halos(h):
    assert f'predictor_large_roi4_h{h}' in step10d.MODES
    assert f'predictor_v3_large_roi4_h{h}' in step10d.MODES
    assert f'random_large_roi4_h{h}' in step10d.MODES


@pytest.mark.parametrize('name,version',[
    ('linear_640',None),('full_bicubic_640',None),('full_large_fsrcnn_640',None),
    ('random_large_roi4_h10',None),('predictor_large_roi4_h10','v2'),
    ('predictor_large_roi4_h16','v2'),('predictor_v3_large_roi4_h10','v3'),
    ('predictor_v3_large_roi4_h16','v3'),
])
def test_correct_predictor_version(name,version):
    assert step10d.scorer_version(name)==version


def test_unknown_predictor_method_rejected():
    with pytest.raises(ValueError):step10d.scorer_version('predictor_v3_large_roi4_h100')


def test_protocol_valid():
    step10d.validate_protocol(*pconfig())


@pytest.mark.parametrize('field,value', [('device','1'),('imgsz',320),('confidence',.4),('nms_iou',.8),('feather_hr_px',0)])
def test_frozen_settings_protected(field,value):
    a,b,c=pconfig();b[field]=value
    with pytest.raises(ValueError):step10d.validate_protocol(a,b,c)


def test_manifest_size_protected():
    a,b,c=pconfig();c['selected_images']=199
    with pytest.raises(ValueError):step10d.validate_protocol(a,b,c)


def test_v2_v3_tile_selection_and_halo_consistent():
    scores=np.arange(16,dtype=np.float64)
    name='video-a-frame-000001-test.png'
    for ver in ('predictor','predictor_v3'):
        a=step10d.assigned_tiles(f'{ver}_large_roi4_h10',name,scores,143)
        b=step10d.assigned_tiles(f'{ver}_large_roi4_h16',name,scores,143)
        assert a==b==[15,14,13,12]


def test_v3_render_uses_halo(monkeypatch):
    seen=[]
    def fake(*args,**kwargs):
        seen.append(kwargs['halo'])
        return np.zeros((512,640),np.uint8),33600
    monkeypatch.setattr(step10d,'render_batched_fsrcnn',fake)
    gray=np.zeros((256,320),np.uint8)
    for h in (10,16):
        image,count=step10d.render(f'predictor_v3_large_roi4_h{h}',gray,[],[0,1,2,3],None,None)
        assert image.shape==(512,640) and count==33600
    assert seen==[10,16]


def test_selection_without_gt():
    a=step10d.assigned_tiles('random_large_roi4_h10','abc',None,143)
    b=step10d.assigned_tiles('random_large_roi4_h16','abc',None,143)
    assert a==b and len(a)==4


def test_utility_sign():
    assert step10d.utility_delta({'small_tp':8,'fp':4},{'small_tp':6,'fp':8})==3


def test_cluster_bootstrap_constant():
    by_seq={'a':['a1','a2'],'b':['b1'],'c':['c1']}
    vals={k:.25 for group in by_seq.values() for k in group}
    assert step10d.cluster_bootstrap_ci(vals,by_seq,143,200)==[.25,.25]


def test_synced_trial_uses_correct_scorer(monkeypatch):
    selections=[]
    monkeypatch.setattr(step10d,'render',lambda *args:(np.zeros((512,640),np.uint8),1234))
    monkeypatch.setattr(step10d,'prediction_records',lambda *args:[])
    monkeypatch.setattr(step10d.torch.cuda,'reset_peak_memory_stats',lambda *args:None)
    monkeypatch.setattr(step10d.torch.cuda,'max_memory_allocated',lambda *args:0)
    import torch
    cfg={'seed':143}
    device=torch.device('cpu')
    class Scorer:
        def __init__(self,offset):self.offset=offset
        def __call__(self,gray):
            selections.append(self.offset)
            return np.arange(16,dtype=float)*self.offset
    scorers={'v2':Scorer(1),'v3':Scorer(-1)}
    name='video-a-frame-000001-test.png'
    for method in ('predictor_large_roi4_h10','predictor_v3_large_roi4_h10'):
        expected_tiles=step10d.assigned_tiles(method,name,scorers[step10d.scorer_version(method)](None),143)
        step10d.timed_trial(method,np.zeros((256,320),np.uint8),name,[],scorers,None,None,cfg,device,
                             {method:tuple(expected_tiles)})
    assert selections==[1,1,-1,-1]


def test_summarize_compares_v3_to_v2_and_bicubic():
    names=['video-a-frame-001000-z.png','video-b-frame-002000-z.png']
    det=[];times=[]
    for img in names:
        for m in step10d.MODES:
            tp=7 if m=='predictor_v3_large_roi4_h10' else (6 if m=='predictor_large_roi4_h10' else 5)
            det.append({'image':img,'sequence':step10d.sequence_id(img),'method':m,'small_gt':10,'small_tp':tp,'all_tp':tp,'fp':2})
            for k in range(3):
                times.append({'image':img,'sequence':step10d.sequence_id(img),'method':m,'repeat':k,
                              'selector_ms':1,'sr_ms':2,'detector_ms':9,'total_ms':12,'computed_pixels':43008})
    result=step10d.summarize(times,det,143)
    comp=result['comparisons']['predictor_v3_large_roi4_h10__vs__predictor_large_roi4_h10']
    assert comp['delta_small_tp']==2
    assert comp['delta_utility_total']==2
    assert comp['utility_per_image_sequence_bootstrap_95ci']==[1.,1.]


def test_config_links_both_checkpoints_and_csvs():
    d=yaml.safe_load((ROOT/'configs'/'large_v2_v3_step10d.yaml').read_text())
    assert d['predictor_v2_checkpoint'].endswith('roi_utility_v2.pt')
    assert d['predictor_v3_checkpoint'].endswith('roi_utility_v3_large_h10.pt')
    assert d['predictor_v3_training_csv'].endswith('large_sr_labels_300_step10a_v1/roi_utility.csv')
    assert d['frozen_manifest']=='reports/development/split_200_v2/manifest.json'
    assert d['expected_sequences']==17
