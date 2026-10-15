"""CPU-only tests of Step11K frozen test preparation and no-overwrite checks."""
import copy
import csv
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
j=importlib.import_module('audit_flir_test_integrity_step11j')
m=importlib.import_module('prepare_frozen_flir_test_step11k')
CFG=yaml.safe_load((ROOT/'configs/flir_frozen_test_preparation_step11k.yaml').read_text())
JCFG=yaml.safe_load((ROOT/'configs/flir_test_integrity_step11j.yaml').read_text())


def one_image(index=1, basename=None, vid='abc'):
    return {'id':index,'file_name':f'data/{basename or f"video-{vid}-frame-000001-z{index}.jpg"}',
            'width':640,'height':512,'extra_info':{'video_id':vid}}


def one_ann(image_id=1,cat=3,box=None):
    return {'image_id':image_id,'category_id':cat,'bbox':box or [10,20,30,40]}


def test_frozen_config_valid():
    m.checked_config(CFG)

@pytest.mark.parametrize('key,value',[
    ('source_split','images_thermal_val'),('source_type','raw_tiff'),
    ('lr_preprocessing','cv2_cubic'),('expected_frames',3748),
    ('expected_videos',9),('expected_coco_annotations',62316),
    ('expected_kept_boxes',62014),
])
def test_reject_changed_fixed_settings(key,value):
    cfg=copy.deepcopy(CFG);cfg[key]=value
    with pytest.raises(ValueError,match='Frozen Step11K'):
        m.checked_config(cfg)

@pytest.mark.parametrize('sha',[None,'x'*64,'a'*63,'bad'])
def test_reject_malformed_expected_digest(sha):
    cfg=copy.deepcopy(CFG);cfg['expected_test_coco_sha256']=sha
    with pytest.raises(ValueError,match='SHA256'):
        m.checked_config(cfg)


def test_yolo_label_positive_roundtrip(tmp_path):
    rows=[(2,.5,.123456789,.1,.333333333),(0,.1,.2,.3,.4)]
    target=tmp_path/'labels.txt'
    target.write_text(m.format_yolo(rows),encoding='utf-8')
    assert j.match_label_sets(rows,j.read_yolo(target),tol=2e-8)[0]
    assert target.read_text().splitlines()[0].startswith('2 ')


def test_empty_yolo_label_is_empty_file():
    assert m.format_yolo([])==''

@pytest.mark.parametrize('cls,coord',[(9,0.5),(-1,0.5),(0,float('nan')),(0,-.2),(0,1.1)])
def test_reject_invalid_yolo_content(cls,coord):
    with pytest.raises(ValueError):m.format_yolo([(cls,coord,.2,.3,.4)])


def test_index_source_counts_excluded_and_empty():
    imgs=[one_image(1),one_image(2,vid='other')]
    anns=[one_ann(1,3),one_ann(1,11),one_ann(2,1)]
    images,idx,by,seqs,total=m.index_sources(imgs,anns,j.validate_config(JCFG)[1])
    assert len(images)==2 and seqs=={'abc','other'} and total==2
    assert len(by[1])==1 and len(by[2])==1
    assert idx[1][1]=='video-abc-frame-000001-z1'


def test_empty_unannotated_image_preserved():
    imgs=[one_image(1),one_image(2)]
    out,_,by,_,total=m.index_sources(imgs,[one_ann(1)],j.validate_config(JCFG)[1])
    assert len(out)==2 and total==1 and by.get(2,[])==[]

@pytest.mark.parametrize('name',[
    'test.jpg','data/../../unsafe.jpg','test/xyz.jpg',
    'data/f.png','data\\a.jpg','data/nested/a.jpg',
])
def test_reject_unexpected_test_filename(name):
    with pytest.raises(ValueError):
        m.index_sources([one_image(basename=name) if name.startswith('data/') else
                         {'id':1,'file_name':name,'width':640,'height':512,
                          'extra_info':{'video_id':'v'}}],[],j.validate_config(JCFG)[1])


def test_reject_duplicate_stems():
    img1=one_image(1,basename='video-abc-frame-00001-a.jpg')
    img2=one_image(2,basename='video-abc-frame-00001-a.jpg')
    with pytest.raises(ValueError):m.index_sources([img1,img2],[],j.validate_config(JCFG)[1])


def test_reject_unexpected_image_dimensions():
    image=one_image();image['height']=200
    with pytest.raises(ValueError):m.index_sources([image],[],j.validate_config(JCFG)[1])


def test_sha_content_changes():
    row={k:'example' for k in m.CSV_FIELDS}
    s1=m.manifest_digest([row]);row['lr_png_sha256']='DIFFERENT'
    assert m.manifest_digest([row])!=s1 and len(s1)==64


def make_prepared(tmp_path):
    data=tmp_path/'data_out';report=tmp_path/'report_out'
    (data/'images'/'test').mkdir(parents=True)
    (data/'labels'/'test').mkdir(parents=True)
    Image.fromarray(np.zeros((256,320),dtype=np.uint8)).save(data/'images'/'test'/'one.png')
    (data/'labels'/'test'/'one.txt').write_text('',encoding='utf-8')
    record={'image':'one.png','video_id':'foo','coco_image_id':'7',
            'raw_jpg_sha256':'0'*64,'lr_png_sha256':m.sha256(data/'images'/'test'/'one.png'),
            'labels_txt_sha256':m.sha256(data/'labels'/'test'/'one.txt'),
            'num_yolo_labels':'0'}
    report.mkdir()
    with (report/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=m.CSV_FIELDS);w.writeheader();w.writerow(record)
    summary={'protocol':m.PROTOCOL,'images':3749,'manifest_content_sha256':m.manifest_digest([record]),
             'data_output':str(data.resolve())}
    (report/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    return data,report,record


def test_verify_outputs_success_for_synthetic_one_frame(tmp_path):
    data,report,_=make_prepared(tmp_path)
    summary=json.loads((report/'summary.json').read_text());summary['images']=1
    (report/'summary.json').write_text(json.dumps(summary))
    assert m.verify_outputs(data,report,expected_images=1)['images']==1


def test_verification_blocks_incomplete_manifest(tmp_path):
    data,report,_=make_prepared(tmp_path)
    with pytest.raises(ValueError,match='Incomplete'):
        m.verify_outputs(data,report)


def test_verification_detects_manifest_checksum_change(tmp_path):
    data,report,record=make_prepared(tmp_path)
    summary=json.loads((report/'summary.json').read_text());summary['images']=1
    (report/'summary.json').write_text(json.dumps(summary))
    with (report/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=m.CSV_FIELDS);w.writeheader()
        changed=dict(record,raw_jpg_sha256='a'*64);w.writerow(changed)
    with pytest.raises(ValueError,match='checksum'):
        m.verify_outputs(data,report,expected_images=1)


def test_verification_detects_modified_png(tmp_path):
    data,report,_=make_prepared(tmp_path)
    summary=json.loads((report/'summary.json').read_text());summary['images']=1
    (report/'summary.json').write_text(json.dumps(summary))
    (data/'images'/'test'/'one.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='changed'):
        m.verify_outputs(data,report,expected_images=1)


def test_verification_detects_label_modification(tmp_path):
    data,report,_=make_prepared(tmp_path)
    summary=json.loads((report/'summary.json').read_text());summary['images']=1
    (report/'summary.json').write_text(json.dumps(summary))
    (data/'labels'/'test'/'one.txt').write_text('0 .5 .5 .1 .1\n')
    with pytest.raises(ValueError,match='changed'):
        m.verify_outputs(data,report,expected_images=1)


def test_verification_detects_output_path_change(tmp_path):
    data,report,_=make_prepared(tmp_path)
    summary=json.loads((report/'summary.json').read_text());summary['images']=1
    summary['data_output']=str(tmp_path/'different')
    (report/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='directory'):
        m.verify_outputs(data,report,expected_images=1)


def test_check_audit_fails_if_mismatched_sha(tmp_path):
    fake=tmp_path/'config.yaml';fake.write_text('not the pinned cfg')
    with pytest.raises(ValueError,match='config SHA256'):
        m.check_audit(CFG,{},fake,JCFG,tmp_path)


def test_no_overwrite_existing_dataset(tmp_path):
    import argparse
    data=tmp_path/'existing';data.mkdir()
    args=argparse.Namespace(config=tmp_path/'missing.yaml',data_out=data,report_out=tmp_path/'new')
    with pytest.raises(FileExistsError,match='Refusing to overwrite'):
        m.prepare(args)


def test_no_overwrite_existing_report(tmp_path):
    import argparse
    report=tmp_path/'existing';report.mkdir()
    args=argparse.Namespace(config=tmp_path/'missing.yaml',data_out=tmp_path/'new',report_out=report)
    with pytest.raises(FileExistsError,match='Refusing to overwrite'):
        m.prepare(args)


def test_no_test_inference_imported():
    import inspect
    source=inspect.getsource(m)
    assert 'from ultralytics import' not in source
    assert 'YOLO(' not in source
    assert 'torch' not in source
