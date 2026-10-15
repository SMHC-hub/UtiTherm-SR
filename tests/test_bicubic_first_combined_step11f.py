"""CPU-only tests: protocol, true Bicubic background, ranking, paired summary."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import json

import cv2
import numpy as np
import pytest
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts'/'evaluate_bicubic_first_combined_step11f.py'
spec=importlib.util.spec_from_file_location('step11f',SCRIPT)
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def config():
    exp=yaml.safe_load((ROOT/'configs'/'bicubic_first_combined_step11f.yaml').read_text())
    base={'imgsz':640,'budget_tiles':4,'feather_hr_px':8,'confidence':.25,'nms_iou':.7,
          'false_positive_penalty':.25,'device':'0'}
    frozen={'protocol':'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST','selected_images':200,'unique_sequences':17}
    return exp,base,frozen


def test_protocol_ok():s.validate_protocol(*config())


@pytest.mark.parametrize('key,val',[('budget_tiles',1),('halo',16),('feather_hr_px',2),('confidence',.5),
                                     ('nms_iou',.3),('false_positive_penalty',1),('expected_images',100),
                                     ('expected_sequences',12)])
def test_protocol_freezes_experiment(key,val):
    a,b,c=config();a[key]=val
    with pytest.raises(ValueError):s.validate_protocol(a,b,c)


@pytest.mark.parametrize('key,val',[('imgsz',320),('budget_tiles',2),('feather_hr_px',0),
                                     ('confidence',.1),('nms_iou',.6),('false_positive_penalty',.5),
                                     ('device','1')])
def test_protocol_freezes_base(key,val):
    a,b,c=config();b[key]=val
    with pytest.raises(ValueError):s.validate_protocol(a,b,c)


def test_protocol_freezes_manifest():
    a,b,c=config();c['unique_sequences']=18
    with pytest.raises(ValueError):s.validate_protocol(a,b,c)


def test_methods_are_unique():assert len(set(s.MODES))==len(s.MODES)==4


def test_zero_budget_reference_selection():
    for mode in s.MODES[:2]:
        assert s.selected_tiles(mode,np.zeros(s.SHAPE,np.uint8),'video-a-frame-3.png',[],42,None)==[]


def test_contrast_tile_std_exact():
    gray=np.arange(256*320,dtype=np.uint32).reshape(s.SHAPE)%256
    gray=gray.astype(np.uint8)
    tiles=[(x*160,y*128,(x+1)*160,(y+1)*128) for y in range(4) for x in range(4)]
    vals=s.contrast_scores(gray,tiles)
    expected=[float(gray.astype(np.float32)[y0//2:y1//2,x0//2:x1//2].std())
              for x0,y0,x1,y1 in tiles]
    assert vals.tolist()==expected


def test_contrast_selection_highest():
    gray=np.zeros(s.SHAPE,dtype=np.uint8)
    tiles=[(x*160,y*128,(x+1)*160,(y+1)*128) for y in range(4) for x in range(4)]
    for i in [2,7,12,15]:
        x0,y0,x1,y1=tiles[i]
        crop=gray[y0//2:y1//2,x0//2:x1//2]
        crop[:,:40]=255
    chosen=s.selected_tiles('contrast4_bicubic_first_h10',gray,'video-a-frame-3.png',tiles,42,None)
    assert chosen==[2,7,12,15]


def test_random_selection_delegates_to_frozen_seed():
    args=[]
    def fake(*a):args.append(a);return [0,4,8,12]
    chosen=s.selected_tiles('random4_bicubic_first_h10',np.zeros(s.SHAPE,np.uint8),
                            'video-x-frame-2.png',[],143,fake)
    assert chosen==[0,4,8,12]
    assert args==[('random4','video-x-frame-2.png',4,16,143)]


def test_random_wrong_count_fails():
    with pytest.raises(ValueError):
        s.selected_tiles('random4_bicubic_first_h10',np.zeros(s.SHAPE,np.uint8),'im',[],143,
                          lambda *args:[0,1,1,4])


def test_tied_scores_stable():assert s.topk(np.zeros(16),4)==[0,1,2,3]


@pytest.mark.parametrize('k',[-1,17])
def test_invalid_budget(k):
    with pytest.raises(ValueError):s.topk(np.zeros(16),k)


def test_invalid_frame_fails():
    with pytest.raises(ValueError):s.contrast_scores(np.zeros((320,256),dtype=np.uint8),[(0,0,160,128)]*16)


def test_render_uses_correct_bicubic_only():
    gray=np.random.default_rng(143).integers(0,256,size=s.SHAPE,dtype=np.uint8)
    frame,pixels=s.render('full_bicubic_640',gray,[],[],None,None,None,None)
    assert np.array_equal(frame,cv2.resize(gray,s.HR,interpolation=cv2.INTER_CUBIC))
    assert pixels==0


def test_render_uses_selected_four_bicubic_first():
    calls=[]
    def combined(lr,tiles,selected,sr,device,**kw):
        calls.append((selected,kw))
        return np.zeros((512,640),np.uint8),33600
    frame,pixels=s.render('contrast4_bicubic_first_h10',np.zeros(s.SHAPE,np.uint8),[],
                           [0,1,2,3],None,None,combined,None)
    assert frame.shape==(512,640) and pixels==33600
    assert calls==[([0,1,2,3],{'feather':8,'halo':10})]


def test_full_large_uses_existing_renderer():
    calls=[]
    def full(*a):calls.append(a[-1]);return np.zeros((512,640),np.uint8),81920
    _,pixels=s.render('full_large_fsrcnn_640',np.zeros(s.SHAPE,np.uint8),[],[],None,None,None,full)
    assert pixels==81920 and calls==['full']


def test_combined_requires_four_tiles():
    with pytest.raises(ValueError):s.render('contrast4_bicubic_first_h10',np.zeros(s.SHAPE,np.uint8),[],[1],None,None,None,None)


def test_sequence_parser_rejects_unsafe():
    for x in ('../video-a-frame-x.png','video-a-frame-x.jpg','frame-x.png'):
        with pytest.raises(ValueError):s.seq_of(x)


def test_cluster_constant():
    u={'im1':0.5,'im2':0.5,'im3':0.5}
    assert s.cluster_ci(u,{'s1':['im1','im2'],'s2':['im3']},143,100)==[.5,.5]


def sample_data():
    det=[];tim=[]
    for im in ['video-a-frame-1234-x.png','video-b-frame-1234-y.png','video-c-frame-1234-z.png']:
        for m in s.MODES:
            tp=7 if m=='contrast4_bicubic_first_h10' else (6 if m=='full_bicubic_640' else 5)
            fp=4 if m=='contrast4_bicubic_first_h10' else 3
            det.append({'image':im,'sequence':s.seq_of(im),'method':m,
                        'small_gt':10,'small_tp':tp,'all_tp':tp,'fp':fp})
            for rep in range(3):
                tim.append({'image':im,'sequence':s.seq_of(im),'method':m,'repeat':rep,
                            'selector_ms':1.,'sr_ms':2.,'detector_ms':10.,'total_ms':13.,
                            'computed_pixels':33600.,'peak_torch_mib':80.})
    return det,tim


def test_summary_paired_deltas():
    det,tim=sample_data()
    r=s.summarize(det,tim,3,.25,143,100)
    comp=r['comparisons']['contrast4_bicubic_first_h10__vs__full_bicubic_640']
    assert comp['delta_small_tp']==3
    assert comp['delta_fp']==3
    assert comp['delta_utility_total']==2.25
    assert comp['sequence_bootstrap_utility_per_image_95ci']==[.75,.75]
    assert r['methods']['contrast4_bicubic_first_h10']['small_recall_conf_025']==.7


def test_summary_rejects_missing_rows():
    det,tim=sample_data()
    with pytest.raises(ValueError):s.summarize(det[:-1],tim,3,.25,143,100)


def test_summary_rejects_duplicate_timing():
    det,tim=sample_data()
    with pytest.raises(ValueError):s.summarize(det,tim+[tim[0]],3,.25,143,100)


def test_summary_rejects_mismatched_gt():
    det,tim=sample_data();det[0]['small_gt']=20
    with pytest.raises(ValueError):s.summarize(det,tim,3,.25,143,100)


def test_reference_accepts_matching_controls():
    det,tim=sample_data();r=s.summarize(det,tim,3,.25,143,100)
    ref={'protocol':'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST',
         'results':{'images':3,'unique_sequences':3,'methods':r['methods']}}
    s.verify_reference(r,ref)


def test_reference_rejects_changed_control():
    det,tim=sample_data();r=s.summarize(det,tim,3,.25,143,100)
    ref={'protocol':'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST',
         'results':{'images':3,'unique_sequences':3,
                    'methods':json.loads(json.dumps(r['methods']))}}
    ref['results']['methods']['full_bicubic_640']['fp']+=1
    with pytest.raises(ValueError):s.verify_reference(r,ref)


def test_write_csv_header(tmp_path):
    p=tmp_path/'test.csv';s.write_csv(p,[{'a':1,'b':2}]);assert p.read_text().startswith('a,b')
    with pytest.raises(ValueError):s.write_csv(p,[{'a':1,'b':2}])


def test_timed_trial_includes_selector(monkeypatch):
    monkeypatch.setattr(s.time,'perf_counter_ns',iter(range(100_000_000,106_000_000,1_000_000)).__next__)
    seen=[]
    def chooser(*a):seen.append('choose');return [0,1,2,3]
    def combined(*args,**kwargs):seen.append('render');return np.zeros((512,640),np.uint8),33600
    def predict(*args):seen.append('predict');return []
    _,pred=s.timed_trial('random4_bicubic_first_h10',np.zeros(s.SHAPE,np.uint8),
                         'video-a-frame-1234-x.png',[],143,None,None,{},torch.device('cpu'),
                         chooser,combined,None,predict,{'random4_bicubic_first_h10':(0,1,2,3)})
    assert seen==['choose','render','predict'] and pred==[]


def test_smoke_config_uses_full_bicubic():
    exp,*_=config()
    assert exp['reference_step10d_summary'].endswith('step10d_full_200_v1/summary.json')
    assert exp['large_sr_checkpoint'].endswith('fsrcnn_large_x2_stable_step9d3_v1.pt')


def provenance_fixture():
    hashes={'predictor_checkpoint':'a','detector_checkpoint':'b','sr_checkpoint':'c'}
    ref={'protocol':'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST',
         'split_manifest_sha256':'split',
         'evaluated_artifacts':{'large_sr_checkpoint':{'sha256':'large'}},
         'frozen_checkpoints':{k:{'sha256':v} for k,v in hashes.items()}}
    return ref,hashes


def test_reference_provenance_ok():
    ref,hashes=provenance_fixture()
    s.verify_reference_provenance(ref,'split','large',hashes)


@pytest.mark.parametrize('field',["split_manifest_sha256","large_sr_checkpoint","detector_checkpoint"])
def test_reference_provenance_fails_any_change(field):
    ref,hashes=provenance_fixture()
    if field=='split_manifest_sha256':ref[field]='wrong'
    elif field=='large_sr_checkpoint':ref['evaluated_artifacts'][field]['sha256']='wrong'
    else:ref['frozen_checkpoints'][field]['sha256']='wrong'
    with pytest.raises(ValueError):s.verify_reference_provenance(ref,'split','large',hashes)
