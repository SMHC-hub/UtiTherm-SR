"""CPU-only Step11G deterministic protocol and comparison tests."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('step11g',ROOT/'scripts/evaluate_bicubic_first_deterministic_step11g.py')
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def sample():
    rows=[]; hashes=[]
    for im in ('video-a-frame-001.png','video-b-frame-002.png'):
        for m in s.prev.MODES:
            rows.append({'image':im,'method':m,'sequence':s.prev.seq_of(im),'small_gt':5,
                         'small_tp':3,'all_tp':4,'fp':2})
            hashes.append({'image':im,'method':m,'sequence':s.prev.seq_of(im),
                           'selected_tiles':'[]','rendered_sha256':'a'*64,
                           'predictions_sha256':'b'*64,'num_predictions':6})
    config={'protocol':s.PROTOCOL, 'config_sha256':'a', 'base_config_sha256':'b',
            'manifest_sha256':'c','large_sr_sha256':'d','detector_sha256':'e',
            'frozen_checkpoint_sha256':{'k':'q'}, 'environment':{'p':'x'},
            'images':2,'sequences':2,'repeats_per_image':3,'methods':list(s.prev.MODES),
            'deterministic_settings':{'enabled':True},'repeats_verified':True}
    return config,rows,hashes


def test_expected_methods_and_config():
    cfg=yaml.safe_load((ROOT/'configs/bicubic_first_deterministic_step11g.yaml').read_text())
    base={'imgsz':640,'budget_tiles':4,'feather_hr_px':8,'confidence':.25,'nms_iou':.7,
          'false_positive_penalty':.25,'device':'0'}
    frozen={'protocol':'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST','selected_images':200,'unique_sequences':17}
    s.prev.validate_protocol(cfg,base,frozen)
    assert cfg['methods']==list(s.prev.MODES)


def test_missing_cublas_fails():
    with pytest.raises(RuntimeError,match='CUBLAS_WORKSPACE_CONFIG'):
        s.setup_determinism({})

@pytest.mark.parametrize('env',[{ 'CUBLAS_WORKSPACE_CONFIG':':16:8'},
                                {'CUBLAS_WORKSPACE_CONFIG':'4096:8'},
                                {'CUBLAS_WORKSPACE_CONFIG':':0:0'}])
def test_wrong_cublas_fails(env):
    with pytest.raises(RuntimeError,match='CUBLAS_WORKSPACE_CONFIG'):
        s.setup_determinism(env)


def test_setup_determinism_supported_cpu(monkeypatch):
    import torch, cv2
    old=(torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,
         torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32,
         torch.are_deterministic_algorithms_enabled(),cv2.getNumThreads())
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    try:
        flags=s.setup_determinism({'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
        assert flags['cudnn_deterministic']
        assert not flags['matmul_allow_tf32']
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.backends.cudnn.benchmark=old[0]
        torch.backends.cudnn.deterministic=old[1]
        torch.backends.cudnn.allow_tf32=old[2]
        torch.backends.cuda.matmul.allow_tf32=old[3]
        torch.use_deterministic_algorithms(old[4])
        cv2.setNumThreads(old[5])


def test_setup_rejects_initialized_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    with pytest.raises(RuntimeError,match='initialized'):
        s.setup_determinism({'CUBLAS_WORKSPACE_CONFIG':':4096:8'})


def test_sha256_image():
    im=np.zeros((512,640),np.uint8)
    assert s.image_fingerprint(im)==s.digest_bytes(im.tobytes())
    im[3,4]=1
    assert s.image_fingerprint(im)!=s.image_fingerprint(np.zeros_like(im))

@pytest.mark.parametrize('bad',[np.zeros((512,640,1),np.uint8),
                                  np.zeros((256,320),np.uint8),
                                  np.zeros((512,640),np.float32)])
def test_invalid_image_rejected(bad):
    with pytest.raises(ValueError):s.image_fingerprint(bad)


def test_pred_fingerprint_is_stable_and_sensitive():
    p=[{'box':(1.0,2,3,4),'conf':.25,'cls':0}]
    a=s.prediction_fingerprint(p)
    assert a==s.prediction_fingerprint(p)
    assert a!=s.prediction_fingerprint([{'box':(1.0,2,3,4),'conf':.251,'cls':0}])
    assert a!=s.prediction_fingerprint([])

@pytest.mark.parametrize('bad',[[{'box':(1,2,3,4),'conf':float('nan'),'cls':0}],
                                [{'box':(1,2,3),'conf':.3,'cls':0}],
                                [{'box':(1,2,3,4),'conf':1.5,'cls':0}],
                                [{'other':4}],{}])
def test_bad_predictions_fail(bad):
    with pytest.raises((ValueError,TypeError,KeyError)):s.prediction_fingerprint(bad)


def test_empty_compare_rejected():
    with pytest.raises(ValueError):s.compare_rows([],[],('small_tp',))


def test_equal_rows_are_equal():
    _,r,h=sample()
    assert s.compare_rows(r,[dict(x) for x in r],s.METRICS)==[]
    assert s.compare_rows(h,[dict(x) for x in h],('rendered_sha256',))==[]


def test_csv_string_vs_int_comparison_ok():
    _,r,_=sample()
    converted=[{k:str(v) for k,v in a.items()} for a in r]
    assert s.compare_rows(r,converted,s.METRICS)==[]

@pytest.mark.parametrize('field',[ 'small_tp','fp','all_tp','small_gt'])
def test_detection_change_found(field):
    _,r,_=sample();r2=[dict(x) for x in r]
    r2[0][field]+=1
    mismatch=s.compare_rows(r2,r,s.METRICS)
    assert len(mismatch)==1 and mismatch[0]['field']==field

@pytest.mark.parametrize('field',['selected_tiles','rendered_sha256','predictions_sha256','num_predictions'])
def test_fingerprint_change_found(field):
    _,_,h=sample();h2=[dict(x) for x in h]
    h2[0][field]='changed'
    mismatch=s.compare_rows(h2,h,('selected_tiles','rendered_sha256','predictions_sha256','num_predictions'))
    assert len(mismatch)==1 and mismatch[0]['field']==field


def test_duplicate_reference_fails():
    _,r,_=sample()
    with pytest.raises(ValueError,match='Duplicate'):s.compare_rows(r,r+[r[0]],s.METRICS)


def test_duplicate_current_fails():
    _,r,_=sample()
    with pytest.raises(ValueError,match='Duplicate'):s.compare_rows(r+[r[0]],r,s.METRICS)


def test_missing_image_fails():
    _,r,_=sample()
    with pytest.raises(ValueError,match='Image/method records disagree'):
        s.compare_rows(r[:-1],r,s.METRICS)


def test_perfect_cross_run():
    c,r,h=sample()
    assert s.compare_run(dict(c),list(r),list(h),dict(c),list(r),list(h))==[]

@pytest.mark.parametrize('field',['config_sha256','base_config_sha256','manifest_sha256',
                                   'large_sr_sha256','detector_sha256','frozen_checkpoint_sha256',
                                   'environment','images','sequences','repeats_per_image',
                                   'methods','deterministic_settings'])
def test_mismatched_provenance_fails(field):
    c,r,h=sample();c2=dict(c);c2[field]='wrong'
    with pytest.raises(ValueError,match='provenance/config'):
        s.compare_run(c2,r,h,c,r,h)


def test_historical_protocol_does_not_pass():
    c,r,h=sample();c2=dict(c);c2['protocol']='STEP10D'
    with pytest.raises(ValueError,match='protocol'):
        s.compare_run(c2,r,h,c,r,h)


def test_unverified_reference_fails():
    c,r,h=sample();c2=dict(c);c2['repeats_verified']=False
    with pytest.raises(ValueError,match='internally verified'):
        s.compare_run(c,r,h,c2,r,h)


def test_load_comparison_files(tmp_path):
    c,r,h=sample();p=tmp_path/'summary.json';p.write_text(json.dumps(c))
    for filename,data in [('detection_per_image.csv',r),('fingerprints_per_image.csv',h)]:
        with (tmp_path/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    c2,r2,h2=s.load_comparison(p)
    assert c2==c and len(r2)==len(r) and len(h2)==len(h)


def test_load_comparison_missing_csv(tmp_path):
    c,_,_=sample();p=tmp_path/'summary.json';p.write_text(json.dumps(c))
    with pytest.raises(FileNotFoundError):s.load_comparison(p)


def test_load_comparison_extra_row(tmp_path):
    c,r,h=sample();p=tmp_path/'summary.json';p.write_text(json.dumps(c))
    for filename,data in [('detection_per_image.csv',r+r[:1]),('fingerprints_per_image.csv',h)]:
        with (tmp_path/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    with pytest.raises(ValueError,match='completeness'):s.load_comparison(p)


def test_class_uses_original_step11f_methods():
    import inspect
    text=inspect.getsource(s.run)
    assert 'prev.validate_protocol' in text
    assert 'prev.verify_reference_provenance' in text
    assert 'prev.summarize' in text
    assert 'render_bicubic_first_fsrcnn' in text
    assert 'repeat_mismatches' in text
    assert 'cross_diffs' in text
    assert 'prev.verify_reference(stats,ref)' not in text


def test_no_override_step11f():
    assert (ROOT/'scripts/evaluate_bicubic_first_combined_step11f.py').exists()


def test_base_recall_count_function_threshold():
    import inspect
    code=inspect.getsource(s.run)
    assert 'detection_counts(gts,first[mode],float(base[\'confidence\']))' in code
