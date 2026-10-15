"""CPU tests for Step 9G experiment protocol and comparison math."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_large_fsrcnn_step9g as step9g  # noqa: E402


def protocol():
    exp = {'halos':[10,16], 'budget_tiles':4, 'false_positive_penalty':0.25,
           'expected_images':200, 'expected_sequences':17}
    original={'imgsz':640,'budget_tiles':4,'feather_hr_px':8,'confidence':.25,
              'nms_iou':.7,'false_positive_penalty':.25,'device':'0'}
    manifest={'selected_images':200, 'unique_sequences':17}
    return exp, original, manifest


def test_modes_all_unique():
    assert len(step9g.MODES) == len(set(step9g.MODES)) == 7


def test_controls_present():
    assert {'linear_640','full_bicubic_640','full_large_fsrcnn_640'} <= set(step9g.MODES)


def test_two_halos_each_have_random_and_predictor():
    for halo in (10,16):
        assert f'random_large_roi4_h{halo}' in step9g.MODES
        assert f'predictor_large_roi4_h{halo}' in step9g.MODES


def test_protocol_valid():
    step9g.validate_protocol(*protocol())


@pytest.mark.parametrize('part,val', [
    ('imgsz',320), ('budget_tiles',8), ('feather_hr_px',3),
    ('confidence',.3), ('nms_iou',.6), ('false_positive_penalty',.4), ('device','1')])
def test_protocol_rejects_modified_detector_settings(part,val):
    a,b,c=protocol()
    b[part]=val
    with pytest.raises(ValueError): step9g.validate_protocol(a,b,c)


@pytest.mark.parametrize('field,new', [('halos',[10]), ('budget_tiles',5), ('false_positive_penalty',.1),('expected_images',199),('expected_sequences',10)])
def test_protocol_rejects_modified_experiment(field,new):
    a,b,c=protocol()
    a[field]=new
    with pytest.raises(ValueError): step9g.validate_protocol(a,b,c)


def test_assigned_tiles_consistent_between_halos():
    scores = np.arange(16, dtype=np.float32)
    img='video-test-frame-000001-something.png'
    a=step9g.assigned_tiles('random_large_roi4_h10',img,None,42)
    b=step9g.assigned_tiles('random_large_roi4_h16',img,None,42)
    assert a == b and len(a) == 4 and len(set(a)) == 4
    a=step9g.assigned_tiles('predictor_large_roi4_h10',img,scores,42)
    b=step9g.assigned_tiles('predictor_large_roi4_h16',img,scores,42)
    assert a == b == [15,14,13,12]


def test_render_passes_correct_halo(monkeypatch):
    calls=[]
    monkeypatch.setattr(step9g, 'render_batched_fsrcnn',
                        lambda *args,**kwargs: (calls.append(kwargs['halo']) or (np.zeros((512,640),np.uint8), 123)))
    for halo in (10,16):
        image,count=step9g.render(f'predictor_large_roi4_h{halo}',np.zeros((256,320),np.uint8),
                                 [(0,0,160,128)]*16,[0,1,2,3],None,None)
        assert image.shape==(512,640) and count==123
    assert calls==[10,16]


def test_render_full_uses_full_model(monkeypatch):
    called=[]
    monkeypatch.setattr(step9g, 'render_fsrcnn',
                        lambda *a,**k: (called.append(a[-1]) or (np.zeros((512,640),np.uint8), 81920)))
    image,n=step9g.render('full_large_fsrcnn_640',np.zeros((256,320),np.uint8),[],[],None,None)
    assert n==81920 and called==['full']


def test_roi_requires_four_tiles():
    with pytest.raises(ValueError):
        step9g.render('random_large_roi4_h10',np.zeros((256,320),np.uint8),[],[1,2,3],None,None)


def test_utility_delta_sign():
    a={'small_tp':11,'fp':5}; b={'small_tp':10,'fp':9}
    assert step9g.utility_delta(a,b)==2.0
    assert step9g.utility_delta(b,a)==-2.0


def test_cluster_ci_constant_values():
    groups={'video-a':['a','b'],'video-b':['c','d'],'video-c':['e']}
    values={i:0.5 for ids in groups.values() for i in ids}
    assert step9g.cluster_bootstrap_ci(values,groups,42,300)==[.5,.5]


def test_cluster_ci_rejects_incomplete_image_list():
    groups={'video-a':['a'],'video-b':['b']}
    with pytest.raises(ValueError):
        step9g.cluster_bootstrap_ci({'a':1},groups,42)


def test_summarize_synthetic_paired_results():
    ims=['video-a-frame-000001-a.png','video-a-frame-000002-b.png',
         'video-b-frame-000001-c.png','video-b-frame-000002-d.png']
    det=[]; trials=[]
    for im in ims:
        for m in step9g.MODES:
            boost=1 if m.startswith('predictor') else 0
            det.append({'image':im,'sequence':step9g.sequence_id(im),'method':m,
                        'small_gt':10,'small_tp':5+boost,'all_tp':5+boost,'fp':2})
            for r in range(3):
                trials.append({'image':im,'sequence':step9g.sequence_id(im),
                               'method':m,'repeat':r,'total_ms':12.0,
                               'selector_ms':1.0,'sr_ms':2.0,'detector_ms':9.0,
                               'computed_pixels':43008})
    result=step9g.summarize(trials,det,42)
    assert result['images']==4 and result['unique_sequences']==2
    comp=result['comparisons']['predictor_large_roi4_h10__vs__random_large_roi4_h10']
    assert comp['delta_small_tp']==4
    assert comp['delta_fp']==0
    assert comp['delta_utility_per_image']==1.0
    assert comp['utility_per_image_sequence_bootstrap_95ci']==[1.,1.]


def test_summarize_rejects_missing_method():
    with pytest.raises(ValueError): step9g.summarize([{}], [], 42)


def test_config_has_new_and_original_path_without_model_overwrite():
    import yaml
    conf=yaml.safe_load((ROOT/'configs'/'large_fsrcnn_step9g.yaml').read_text())
    assert conf['base_development_config']=='configs/development_eval.yaml'
    assert conf['large_sr_checkpoint'].endswith('fsrcnn_large_x2_stable_step9d3_v1.pt')
    assert conf['predictor_v2_checkpoint'].endswith('roi_utility_v2.pt')
    assert conf['expected_sequences']==17
