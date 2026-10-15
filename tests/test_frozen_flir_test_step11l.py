"""Step11L CPU safeguards. No thermal TEST inference in these tests."""
from __future__ import annotations
import importlib.util
import json
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('step11l',ROOT/'scripts/evaluate_frozen_flir_test_step11l.py')
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def base_config():
    c=yaml.safe_load((ROOT/'configs/flir_disjoint_test_evaluation_step11l.yaml').read_text())
    base={'seed':42,'confidence':.25,'nms_iou':.7,'imgsz':640,'false_positive_penalty':.25,'device':'0'}
    g={'methods':list(s.METHODS),'halo':10,'feather_hr_px':8}
    k=yaml.safe_load((ROOT/'configs/flir_frozen_test_preparation_step11k.yaml').read_text())
    return c,base,g,k


def test_config_pass():
    assert s.validate_frozen_config(*base_config())

@pytest.mark.parametrize('field,value',[
    ('expected_test_images',3748),('expected_test_sequences',7),('expected_test_boxes',62016),
    ('budget_tiles',3),('halo',16),('feather_hr_px',4),('reference_prediction_confidence',.4),
    ('nms_iou',.5),('matching_iou',.4),('fp_penalty',0),('random_seed',1),
    ('bootstrap_seed',1),('bootstrap_iterations',100),
    ('expected_untouched_test_manifest_sha256','0'*64),
    ('expected_step11k_config_sha256','0'*64),('expected_source_coco_sha256','0'*64),
    ('primary_comparison','contrast4_at_0.40_vs_bicubic_at_0.40'),
    ('methods',['full_bicubic_640']),
    ('confidence_thresholds',[.25,.35]),
    ('secondary_comparisons',[]),
])
def test_reject_modified_test_protocol(field,value):
    c,base,g,k=base_config();c[field]=value
    with pytest.raises(ValueError):s.validate_frozen_config(c,base,g,k)

@pytest.mark.parametrize('field,value',[
    ('confidence',.30),('nms_iou',.5),('seed',123),('device','cpu'),('false_positive_penalty',.1),('imgsz',512)
])
def test_reject_modified_detector(field,value):
    c,b,g,k=base_config();b[field]=value
    with pytest.raises(ValueError):s.validate_frozen_config(c,b,g,k)

@pytest.mark.parametrize('field,value',[('halo',16),('feather_hr_px',0),('methods',['random4'])])
def test_reject_modified_step11g_method(field,value):
    c,b,g,k=base_config();g[field]=value
    with pytest.raises(ValueError):s.validate_frozen_config(c,b,g,k)


def manifest():
    r=[]
    for i in range(3749):
        vid=f'video-id{i%8}'
        r.append({'image':f'{vid}-frame-{i:06d}.png','video_id':f'id{i%8}',
                  'num_yolo_labels':str(62015//3749+ (1 if i<(62015%3749) else 0))})
    return r


def test_valid_manifest():
    c,*_=base_config()
    source={'protocol':s.step11k.PROTOCOL,'images':3749,'sequences':8,
            'converted_yolo_boxes':62015,
            'manifest_content_sha256':c['expected_untouched_test_manifest_sha256'],
            'config_sha256':c['expected_step11k_config_sha256'],
            'source_coco_sha256':c['expected_source_coco_sha256'],
            'detector_inference_performed':False}
    assert s.validate_sealed_test(c,source,manifest(),True)

@pytest.mark.parametrize('field,value',[
    ('images',3748),('sequences',7),('converted_yolo_boxes',999),
    ('manifest_content_sha256','0'*64),('config_sha256','0'*64),
    ('source_coco_sha256','0'*64),('detector_inference_performed',True),
    ('protocol','BAD')])
def test_reject_source_tamper(field,value):
    c,*_=base_config()
    source={'protocol':s.step11k.PROTOCOL,'images':3749,'sequences':8,
            'converted_yolo_boxes':62015,
            'manifest_content_sha256':c['expected_untouched_test_manifest_sha256'],
            'config_sha256':c['expected_step11k_config_sha256'],
            'source_coco_sha256':c['expected_source_coco_sha256'],
            'detector_inference_performed':False}
    source[field]=value
    with pytest.raises(ValueError):s.validate_sealed_test(c,source,manifest(),True)


def test_reject_duplicate_manifest():
    c,*_=base_config(); r=manifest();r[-1]=dict(r[0]);
    source={'protocol':s.step11k.PROTOCOL,'images':3749,'sequences':8,
            'converted_yolo_boxes':62015,
            'manifest_content_sha256':c['expected_untouched_test_manifest_sha256'],
            'config_sha256':c['expected_step11k_config_sha256'],
            'source_coco_sha256':c['expected_source_coco_sha256'],
            'detector_inference_performed':False}
    with pytest.raises(ValueError):s.validate_sealed_test(c,source,r,True)


def test_smoke_covers_every_video():
    full=manifest()
    smoke=s.select_test_rows(full,8)
    assert len(smoke)==8
    assert len({r['video_id'] for r in smoke})==8
    assert len(s.select_test_rows(full,0))==3749
    assert [r['image'] for r in smoke]==sorted(r['image'] for r in smoke)

@pytest.mark.parametrize('limit',[-1,1,2,7])
def test_reject_bad_limit(limit):
    with pytest.raises(ValueError):s.select_test_rows(manifest(),limit)


def tiny_data():
    rows=[];times=[];finger=[]
    for i in range(8):
        image=f'video-id{i}-frame-{i:04d}.png'
        for m in s.METHODS:
            finger.append({'image':image,'sequence':f'video-id{i}','method':m,'selected_tiles':'[0,1,2,3]',
                           'rendered_sha256':'a'*64,'predictions_sha256':'b'*64,'num_predictions':20})
            for rep in range(3):
                times.append({'image':image,'sequence':f'video-id{i}','method':m,
                              'repeat':rep,'total_ms':10+rep})
            for t in s.CONFIDENCES:
                baseline=12 if m!='contrast4_bicubic_first_h10' else 13
                rows.append({'image':image,'sequence':f'video-id{i}','method':m,
                             'threshold':f'{t:.2f}','small_gt':20,
                             'small_tp':baseline-(2 if t==.40 else 0),
                             'all_tp':baseline+1-(2 if t==.40 else 0),
                             'fp':5-(2 if t==.40 else 0)})
    return rows,finger,times


def test_validate_records_and_compare_stats():
    rows,finger,times=tiny_data()
    assert s.validate_per_image(rows,finger,times,8,3)
    stats=s.summarize(rows,times,.25,143,1000)
    p=stats['predeclared_comparisons']['primary_contrast025_vs_bicubic025']
    assert p['delta_small_tp']==8 and p['delta_fp']==0 and p['delta_utility_total']==8
    assert stats['by_confidence']['0.25']['full_bicubic_640']['absolute_utility']==8*(12-5*.25)
    assert stats['by_confidence']['0.40']['full_bicubic_640']['small_tp']==80
    assert p['sequence_wins_ties_losses']['wins']==8


def test_threshold_monotonic_guard():
    rows,fp,t=tiny_data()
    for r in rows:
        if r['threshold']=='0.40' and r['method']=='full_bicubic_640':
            r['fp']=9;break
    with pytest.raises(ValueError,match='increased counts'):
        s.validate_per_image(rows,fp,t,8,3)

@pytest.mark.parametrize('part',[0,1,2])
def test_duplicate_records_rejected(part):
    rows,fp,t=tiny_data()
    (rows,fp,t)[part].append(dict((rows,fp,t)[part][0]))
    with pytest.raises(RuntimeError):s.validate_per_image(rows,fp,t,8,3)


def run_summary():
    return {'protocol':s.PROTOCOL,'config_sha256':'a','step11g_provenance':{'x':1},
            'step11k_manifest_content_sha256':'b','step11k_summary_sha256':'c',
            'environment':{'python':'3'},'deterministic_settings':{'cuda':1},
            'images':8,'sequences':8,'repeats_per_image':3,
            'methods':list(s.METHODS),'thresholds':list(s.CONFIDENCES),
            'prepared_image_names_sha256':'abc','repeats_verified':True}


def test_cross_run_exact():
    rows,fp,_=tiny_data();a=run_summary()
    assert s.compare_run(a,rows,fp,a,rows,fp)==[]

@pytest.mark.parametrize('field',['config_sha256','step11g_provenance',
    'step11k_manifest_content_sha256','step11k_summary_sha256','environment',
    'deterministic_settings','images','sequences','repeats_per_image','methods',
    'thresholds','prepared_image_names_sha256'])
def test_cross_run_provenance_tamper(field):
    rows,fp,_=tiny_data();a=run_summary();b=copy.deepcopy(a);b[field]='wrong'
    with pytest.raises(ValueError,match='provenance mismatch'):
        s.compare_run(a,rows,fp,b,rows,fp)

@pytest.mark.parametrize('field',['small_gt','small_tp','all_tp','fp'])
def test_cross_run_detection_change(field):
    rows,fp,_=tiny_data();other=copy.deepcopy(rows)
    other[0][field]+=1
    diffs=s.compare_run(run_summary(),other,fp,run_summary(),rows,fp)
    assert len(diffs)==1 and diffs[0]['field']==field

@pytest.mark.parametrize('field',['rendered_sha256','predictions_sha256','selected_tiles','num_predictions'])
def test_cross_run_fingerprint_change(field):
    rows,fp,_=tiny_data();other=copy.deepcopy(fp);other[0][field]='bad'
    diffs=s.compare_run(run_summary(),rows,other,run_summary(),rows,fp)
    assert len(diffs)==1 and diffs[0]['field']==field


def test_reject_comparison_duplicates():
    rows,fp,_=tiny_data(); rows.append(dict(rows[0]))
    with pytest.raises(ValueError,match='Duplicate comparison row'):
        s.compare_run(run_summary(),rows,fp,run_summary(),rows[:-1],fp)


def test_no_previous_project_overwrite():
    for name in ('scripts/evaluate_bicubic_first_deterministic_step11g.py',
                 'scripts/evaluate_bicubic_first_combined_step11f.py',
                 'scripts/prepare_frozen_flir_test_step11k.py'):
        assert (ROOT/name).exists()


def test_no_test_training_or_threshold_optimizer():
    code=(ROOT/'scripts/evaluate_frozen_flir_test_step11l.py').read_text()
    assert 'optimizer.step' not in code and 'model.train(' not in code
    assert 'select_test_rows' in code and 'step11k.verify_outputs' in code
    assert 'step11g.render_detect_trial' in code
    assert 'step11g.setup_determinism' in code


def test_frozen_config_contains_same_protocol_constants():
    c,*_=base_config()
    assert c['primary_comparison']=='contrast4_at_0.25_vs_bicubic_at_0.25'
    assert c['random_seed']==42
    assert c['confidence_thresholds']==[.25,.4]
