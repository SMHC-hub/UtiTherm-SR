"""Step11K: prepare sealed 8-bit FLIR video-test LR/YOLO labels; NO inference.

Requires a PASSED Step11J audit with exact validation preprocessing and label parity.
Produces a brand-new output directory with images/test/*.png, labels/test/*.txt,
plus an immutable-input manifest in a separate new report directory.

Usage from repo root:
  python scripts/prepare_frozen_flir_test_step11k.py
  python scripts/prepare_frozen_flir_test_step11k.py --verify-only
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image
import yaml

# Import exactly the converter whose parity Step11J verified on 1144 validation frames.
from audit_flir_test_integrity_step11j import (
    validate_config, safe_rel_filename, video_id, load_dataset,
    coco_bbox_to_yolo, low_resolution, sha256, read_yolo, match_label_sets,
)

PROTOCOL = 'STEP11K_FROZEN_FLIR_VIDEO_TEST_INPUT_PREPARATION_V1'
CSV_FIELDS = ('image','video_id','coco_image_id','raw_jpg_sha256','lr_png_sha256',
              'labels_txt_sha256','num_yolo_labels')
DEFAULT_DATA = Path('data/processed/yolo_flir_lr_x2_test_step11k_v1')
DEFAULT_REPORT = Path('reports/test/step11k_test_preparation_v1')


def checked_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError('Config must be a mapping')
    expected = {
        'source_split':'video_thermal_test',
        'source_type':'coco_referenced_8bit_jpg',
        'lr_preprocessing':'pil_bicubic',
        'expected_frames':3749,
        'expected_videos':8,
        'expected_coco_annotations':62317,
        'expected_kept_boxes':62015,
    }
    for key, val in expected.items():
        if cfg.get(key) != val:
            raise ValueError(f'Frozen Step11K setting changed: {key} expected {val!r}')
    for key in ('expected_test_coco_sha256','expected_step11j_config_sha256'):
        value = cfg.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Expected SHA256 missing or invalid: {key}')


def check_audit(cfg: dict, audit: dict, jconfig_path: Path, jconfig:dict, raw_root:Path) -> dict[int,int]:
    """Fail closed if Step11J audit, input COCO, or conversion assumptions changed."""
    validate_config(jconfig)
    mapping = {int(k):int(v) for k,v in jconfig['coco_category_to_yolo'].items()}
    if sha256(jconfig_path) != cfg['expected_step11j_config_sha256']:
        raise ValueError('Step11J config SHA256 differs from frozen audit')
    if audit.get('config_sha256') != cfg['expected_step11j_config_sha256']:
        raise ValueError('Step11J audit used different configuration')
    if audit.get('protocol')!='STEP11J_FLIR_DISJOINT_TEST_PREPARATION_READ_ONLY_AUDIT_V1':
        raise ValueError('Unknown Step11J audit protocol')
    if audit.get('ready_for_test_preparation') is not True or audit.get('sequence_disjoint') is not True:
        raise ValueError('Step11J source integrity/parity gate not passed')
    cal = audit.get('validation_image_calibration',{})
    if cal.get('frames',0)<51 or cal.get('exact_image_matches_by_method',{}).get('pil_bicubic') != cal.get('frames'):
        raise ValueError('PIL Bicubic validation pixel parity was not established')
    lc=audit.get('validation_label_calibration',{})
    if lc.get('images')!=1144 or lc.get('matches')!=1144 or lc.get('mismatches')!=0:
        raise ValueError('COCO label mapping not established on all val images')
    if {int(k):int(v) for k,v in audit.get('coco_category_to_yolo',{}).items()} != mapping:
        raise ValueError('Test label mapping differs from successful Step11J audit')
    expected = {'train':(10742,133),'val':(1144,17),'test':(3749,8)}
    for split,(images,seqs) in expected.items():
        entry=audit.get('dataset_splits',{}).get(split,{})
        if entry.get('images')!=images or entry.get('sequences')!=seqs:
            raise ValueError(f'Step11J {split} inventory count mismatch')
        source_folder={'train':'images_thermal_train','val':'images_thermal_val','test':'video_thermal_test'}[split]
        if sha256(raw_root/source_folder/'coco.json')!=entry.get('coco_sha256'):
            raise ValueError(f'{split} COCO metadata changed since Step11J audit')
    if audit['dataset_splits']['test']['coco_sha256']!=cfg['expected_test_coco_sha256']:
        raise ValueError('Test COCO SHA256 differs from pinned preparation input')
    return mapping


def index_sources(images:list[dict],anns:list[dict],mapping:dict[int,int]):
    """Validate all frame identifiers before writing any output."""
    ordered=sorted(images,key=lambda x:Path(str(x['file_name'])).name)
    seen=set(); index={}; per_img=defaultdict(list); seqs=set(); all_anns=0
    for im in ordered:
        rel=safe_rel_filename(im)
        if len(rel.parts)!=2 or rel.parts[0]!='data' or rel.suffix.lower()!='.jpg':
            raise ValueError(f'Unsupported test JPG path: {rel}')
        stem=rel.stem
        if not stem or stem in seen:
            raise ValueError(f'Duplicate test stem: {stem}')
        seen.add(stem)
        if (int(im['width']),int(im['height']))!=(640,512):
            raise ValueError(f'Incorrect COCO dimensions: {rel}')
        vid=video_id(im);seqs.add(vid)
        index[im['id']]=(im,stem,rel,vid)
    for ann in anns:
        im,_,_,_=index[ann['image_id']]
        label=coco_bbox_to_yolo(ann,im,mapping)
        if label is not None:
            per_img[im['id']].append(label)
            all_anns+=1
    return ordered,index,per_img,seqs,all_anns


def format_yolo(rows:list[tuple])->str:
    for cls,*coords in rows:
        if int(cls)!=cls or not 0<=int(cls)<=8 or not all(np.isfinite(q) and 0<=q<=1 for q in coords):
            raise ValueError('Bad YOLO label')
    return ''.join(f'{int(cls)} '+ ' '.join(f'{float(x):.8f}' for x in coords)+'\n' for cls,*coords in rows)


def manifest_digest(rows:list[dict])->str:
    """Stable digest of the ordered manifest rows, independent of CSV newlines."""
    payload=json.dumps([{k:r[k] for k in CSV_FIELDS} for r in rows],sort_keys=True,
                       separators=(',',':'),ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def load_outputs_manifest(report:Path) -> tuple[dict,list[dict]]:
    summary=json.loads((report/'summary.json').read_text(encoding='utf-8'))
    with (report/'manifest.csv').open(newline='',encoding='utf-8') as fh:
        reader=csv.DictReader(fh)
        if tuple(reader.fieldnames or ())!=CSV_FIELDS:
            raise ValueError('Unexpected manifest CSV fields')
        rows=list(reader)
    return summary,rows


def verify_outputs(data:Path,report:Path,*,check_all:bool=True,expected_images:int=3749)->dict:
    summary,rows=load_outputs_manifest(report)
    if summary.get('protocol')!=PROTOCOL:
        raise ValueError('Wrong Step11K report protocol')
    if summary['images']!=len(rows) or summary['images']!=expected_images:
        raise ValueError('Incomplete frozen manifest')
    if manifest_digest(rows)!=summary['manifest_content_sha256']:
        raise ValueError('Frozen manifest checksum mismatch')
    if data.resolve().as_posix()!=Path(summary['data_output']).resolve().as_posix():
        raise ValueError('Prepared output directory differs from manifest')
    for entry in rows if check_all else rows[:8]:
        base=Path(entry['image']).stem
        img=data/'images'/'test'/f'{base}.png'
        label=data/'labels'/'test'/f'{base}.txt'
        if not img.is_file() or not label.is_file():
            raise FileNotFoundError(f'Missing prepared file: {base}')
        if sha256(img)!=entry['lr_png_sha256'] or sha256(label)!=entry['labels_txt_sha256']:
            raise ValueError(f'Prepared output changed since sealing: {base}')
        labels=read_yolo(label)
        if len(labels)!=int(entry['num_yolo_labels']):
            raise ValueError(f'Label count mismatch for {base}')
        if check_all:
            with Image.open(img) as im:
                if im.mode!='L' or im.size!=(320,256):
                    raise ValueError(f'Bad LR PNG shape/mode: {img}')
    if (data/'images'/'test').exists() and len(list((data/'images'/'test').glob('*.png')))!=len(rows):
        raise ValueError('Extra/missing LR images in prepared output')
    if (data/'labels'/'test').exists() and len(list((data/'labels'/'test').glob('*.txt')))!=len(rows):
        raise ValueError('Extra/missing labels in prepared output')
    return summary


def prepare(args)->dict:
    if args.data_out.exists() or args.report_out.exists():
        raise FileExistsError('Refusing to overwrite prior prepared images/labels or report')
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    checked_config(cfg)
    jconfig_path=Path(cfg['step11j_config'])
    jconfig=yaml.safe_load(jconfig_path.read_text(encoding='utf-8'))
    raw_root=Path(jconfig['raw_root'])
    audit_path=Path(cfg['step11j_report'])
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    mapping=check_audit(cfg,audit,jconfig_path,jconfig,raw_root)
    testdir=raw_root/cfg['source_split']
    _,images,anns=load_dataset(testdir/'coco.json')
    ordered,index,by_image,seqs,total=index_sources(images,anns,mapping)
    if len(images)!=cfg['expected_frames'] or len(anns)!=cfg['expected_coco_annotations'] or len(seqs)!=cfg['expected_videos'] or total!=cfg['expected_kept_boxes']:
        raise ValueError(f'Test inventory unexpectedly changed {len(images)} images {len(anns)} annotations {len(seqs)} sequences {total} boxes')
    if not all((testdir/index[im['id']][2]).is_file() for im in ordered):
        raise FileNotFoundError('Missing COCO-referenced test JPG')
    args.data_out.parent.mkdir(parents=True,exist_ok=True)
    args.report_out.parent.mkdir(parents=True,exist_ok=True)
    data_temp=Path(tempfile.mkdtemp(prefix='.step11k_data_incomplete_',dir=args.data_out.parent))
    report_temp=Path(tempfile.mkdtemp(prefix='.step11k_report_incomplete_',dir=args.report_out.parent))
    committed=False
    try:
        (data_temp/'images'/'test').mkdir(parents=True)
        (data_temp/'labels'/'test').mkdir(parents=True)
        manifest=[]
        for i,im in enumerate(ordered,1):
            _,stem,rel,vid=index[im['id']]
            raw_img=testdir/rel
            target_img=data_temp/'images'/'test'/f'{stem}.png'
            target_lbl=data_temp/'labels'/'test'/f'{stem}.txt'
            with Image.open(raw_img) as src:
                lr=low_resolution(src,'pil_bicubic')
            # Target is a single-channel 8-bit PNG; never touch TIFF or source JPEG.
            Image.fromarray(lr,mode='L').save(target_img,format='PNG')
            rows=by_image.get(im['id'],[])
            target_lbl.write_text(format_yolo(rows),encoding='utf-8',newline='\n')
            # Round-trip parity on every frame including empty images.
            ok,details=match_label_sets(rows,read_yolo(target_lbl),tol=0.00000002)
            if not ok:raise RuntimeError(f'Test label roundtrip mismatch: {stem}, {details}')
            with Image.open(target_img) as saved:
                if saved.mode!='L' or saved.size!=(320,256) or not np.array_equal(lr,np.asarray(saved)):
                    raise RuntimeError(f'Lossy or malformed LR PNG: {stem}')
            manifest.append({
                'image':f'{stem}.png','video_id':vid,'coco_image_id':str(im['id']),
                'raw_jpg_sha256':sha256(raw_img),'lr_png_sha256':sha256(target_img),
                'labels_txt_sha256':sha256(target_lbl),'num_yolo_labels':str(len(rows)),
            })
            if i==1 or i%500==0 or i==len(ordered):
                print(f'Frozen FLIR test preparation {i}/{len(ordered)}',flush=True)
        with (report_temp/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=CSV_FIELDS,lineterminator='\n')
            writer.writeheader();writer.writerows(manifest)
        actual_boxes=sum(int(r['num_yolo_labels']) for r in manifest)
        if actual_boxes!=total:raise AssertionError('Converted box total mismatch')
        class_counts=Counter(int(box[0]) for boxes in by_image.values() for box in boxes)
        summary={
            'protocol':PROTOCOL,'created_utc':datetime.now(timezone.utc).isoformat(),
            'source_audit_summary_sha256':sha256(audit_path),
            'source_step11j_config_sha256':sha256(jconfig_path),
            'source_coco_sha256':sha256(testdir/'coco.json'),
            'config_sha256':sha256(args.config),
            'images':len(manifest),'sequences':len(seqs),'annotation_records':len(anns),
            'converted_yolo_boxes':total,'converted_boxes_by_yolo_class':dict(sorted(class_counts.items())),
            'excluded_annotations':len(anns)-total,
            'images_with_empty_yolo_labels':sum(int(r['num_yolo_labels'])==0 for r in manifest),
            'manifest_content_sha256':manifest_digest(manifest),
            'data_output':str(args.data_out.resolve()),
            'report_output':str(args.report_out.resolve()),
            'method':'8-bit thermal JPG grayscale conversion + PIL BICUBIC 640x512->320x256',
            'dimensions':{'input':[640,512],'output':[320,256]},
            'read_only_sources':True,'detector_inference_performed':False,
            'limitations':[
                'No detector predictions or test utility were measured; this is preparation only.',
                'COCO categories other than the nine detector classes remain excluded.',
                'Eight test sequence IDs disjoint from train/val; same FLIR ADAS v2 dataset family.',
                '8-bit COCO JPG source, not radiometric TIFF.',
                'Test set must not inform choice of model, confidence, penalty or selection policy.',
            ],
        }
        (report_temp/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        # No output can be overwritten. Verify every staged artifact before commit.
        staged_summary=dict(summary,data_output=str(data_temp.resolve()))
        (report_temp/'summary.json').write_text(json.dumps(staged_summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        verify_outputs(data_temp,report_temp,check_all=True)
        (report_temp/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        # Atomic directory renames within each filesystem. Both remain new paths.
        if args.data_out.exists() or args.report_out.exists():
            raise FileExistsError('Output appeared during preparation, refusing to replace')
        os.rename(data_temp,args.data_out)
        os.rename(report_temp,args.report_out)
        committed=True
        print('STEP11K — FROZEN DISJOINT TEST PREPARATION; NO MODEL INFERENCE')
        print(f'Prepared {len(manifest)} 320x256 LR PNGs and matching YOLO .txt labels; {total} usable objects across {len(seqs)} videos')
        print('Empty-label frames retained:',summary['images_with_empty_yolo_labels'])
        print('SEALED:',args.data_out)
        print('REPORT:',args.report_out/'summary.json')
        return summary
    finally:
        # Stage cleanup only; never remove raw, prior results, or committed outputs.
        if not committed:
            if data_temp.exists():shutil.rmtree(data_temp)
            if report_temp.exists():shutil.rmtree(report_temp)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/flir_frozen_test_preparation_step11k.yaml'))
    p.add_argument('--data-out',type=Path,default=DEFAULT_DATA)
    p.add_argument('--report-out',type=Path,default=DEFAULT_REPORT)
    p.add_argument('--verify-only',action='store_true',help='Validate sealed outputs without writing')
    args=p.parse_args()
    if args.verify_only:
        summary=verify_outputs(args.data_out,args.report_out)
        print('VERIFIED: prepared test PNGs/labels match frozen manifest for all',summary['images'],'images')
    else:
        prepare(args)

if __name__=='__main__':main()
