"""CPU-only tests for Step11J metadata, category mapping, and parity guardrails."""
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import pytest
import yaml

TARGET=Path(__file__).resolve().parents[1]/'scripts'/'audit_flir_test_integrity_step11j.py'
spec=importlib.util.spec_from_file_location('step11j_integrity',TARGET)
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)
CONFIG=Path(__file__).resolve().parents[1]/'configs'/'flir_test_integrity_step11j.yaml'

@pytest.fixture
def cfg():
    return yaml.safe_load(CONFIG.read_text(encoding='utf-8'))


def test_model_names_and_mapping(cfg):
    names,mapping=m.validate_config(cfg)
    assert names[8]=='sign'
    assert mapping[12]==8
    assert mapping[79]==6
    assert mapping[10]==7
    assert mapping[1]==0

@pytest.mark.parametrize('cat,cls',[(1,0),(2,1),(3,2),(4,3),(6,4),(8,5),(79,6),(10,7),(12,8)])
def test_nine_classes(cfg,cat,cls):
    assert m.validate_config(cfg)[1][cat]==cls

@pytest.mark.parametrize('cat',[5,7,9,11,17,80,999])
def test_excluded_categories(cfg,cat):
    mapping=m.validate_config(cfg)[1]
    assert m.coco_bbox_to_yolo({'category_id':cat,'bbox':[10,20,30,40]},
         {'width':640,'height':512},mapping) is None

@pytest.mark.parametrize('raw',['../escape.jpg','/root/image.jpg','bad.jpg','data/../image.jpg','data\\evil.jpg',''])
def test_unsafe_paths(raw):
    with pytest.raises(ValueError):m.safe_rel_filename({'file_name':raw})


def test_normalize_coco_xywh(cfg):
    row=m.coco_bbox_to_yolo({'category_id':3,'bbox':[100,200,40,20]},
            {'width':640,'height':512},m.validate_config(cfg)[1])
    assert row==(2,120/640,210/512,40/640,20/512)


def test_clip_out_of_bounds(cfg):
    row=m.coco_bbox_to_yolo({'category_id':1,'bbox':[-5,500,25,30]},
            {'width':640,'height':512},m.validate_config(cfg)[1])
    assert row==(0,10/640,506/512,20/640,12/512)


def test_crowd_skipped(cfg):
    assert m.coco_bbox_to_yolo({'category_id':1,'iscrowd':True,'bbox':[1,1,1,1]},
            {'width':640,'height':512},m.validate_config(cfg)[1]) is None


def test_invalid_bbox(cfg):
    with pytest.raises(ValueError):
        m.coco_bbox_to_yolo({'category_id':1,'bbox':[1,1,0,1]},
            {'width':640,'height':512},m.validate_config(cfg)[1])


def test_labels_order_independent():
    a=[(0,.1,.2,.3,.4),(2,.9,.8,.1,.1)]
    assert m.match_label_sets(a,list(reversed(a)))[0]


def test_duplicate_label_multiplicity():
    a=[(0,.1,.2,.3,.4)]*2
    assert not m.match_label_sets(a,a[:1])[0]


def test_label_coordinate_tolerance():
    a=[(0,.1,.2,.3,.4)]
    b=[(0,.10001,.2,.3,.4)]
    c=[(0,.101,.2,.3,.4)]
    assert m.match_label_sets(a,b)[0]
    assert not m.match_label_sets(a,c)[0]


def test_yolo_parsing(tmp_path):
    file=tmp_path/'labels.txt';file.write_text('2 0.5 0.5 0.1 0.1\n',encoding='utf-8')
    assert m.read_yolo(file)==[(2,.5,.5,.1,.1)]

@pytest.mark.parametrize('line',['2 0.2 0.4','99 0.5 0.5 0.1 0.1','2.2 0.5 0.5 0.1 0.1','0 0.2 NaN .2 .2'])
def test_reject_invalid_labels(tmp_path,line):
    file=tmp_path/'bad.txt';file.write_text(line+'\n',encoding='utf-8')
    with pytest.raises(ValueError):m.read_yolo(file)


def test_video_id_from_metadata():
    assert m.video_id({'file_name':'data/video-abc-frame-000001-xyz.jpg',
                       'extra_info':{'video_id':'abc'}})=='abc'


def test_video_id_from_filename():
    assert m.video_id({'file_name':'data/video-abc-frame-000001-xyz.jpg',
                       'extra_info':{}})=='abc'


def test_video_id_reject_unidentified():
    with pytest.raises(ValueError):m.video_id({'file_name':'data/img0002.jpg','extra_info':{}})

@pytest.mark.parametrize('method',m.ALLOWED_METHODS)
def test_candidate_preprocessing(method):
    arr=np.tile(np.arange(640,dtype=np.uint8),(512,1))
    img=Image.fromarray(arr)
    out=m.low_resolution(img,method)
    assert out.shape==(256,320) and out.dtype==np.uint8


def test_sampling_covers_all_sequences():
    images=[{'id':i,'file_name':f'data/video-{i%3}-frame-{i:06d}-x.jpg',
             'extra_info':{'video_id':str(i%3)}} for i in range(30)]
    selected=m.sample_val(images,5)
    assert len(selected)==5 and len({m.video_id(i) for i in selected})==3


def test_sample_clamps_if_fewer_images():
    images=[{'id':i,'file_name':f'data/video-{i}-frame-000-x.jpg',
             'extra_info':{'video_id':str(i)}} for i in range(3)]
    assert len(m.sample_val(images,100))==3


def test_config_rejects_modified_mapping(cfg):
    cfg['coco_category_to_yolo'][12]=7
    with pytest.raises(ValueError):m.validate_config(cfg)


def test_input_dims_guard():
    with pytest.raises(ValueError):m.low_resolution(Image.new('L',(32,32)),'pil_bicubic')


def test_split_invalid_missing_source(tmp_path):
    folder=tmp_path/'test';folder.mkdir()
    import json
    (folder/'coco.json').write_text(json.dumps({'images':[{'id':1,'file_name':'data/image.jpg','width':640,'height':512,'extra_info':{'video_id':'123'}}],'annotations':[]}))
    with pytest.raises(FileNotFoundError):m.inspect_split(folder,1,1)


def test_split_duplicate_names(tmp_path):
    folder=tmp_path/'test';folder.mkdir()
    import json
    (folder/'coco.json').write_text(json.dumps({'images':[{'id':1,'file_name':'data/same.jpg'},{'id':2,'file_name':'data/same.jpg'}],'annotations':[]}))
    with pytest.raises(ValueError):m.load_dataset(folder/'coco.json')
