"""Step11J: read-only FLIR test-split integrity and preprocessing parity audit.

No test detections, no model fitting, no conversion. The 3749-frame test is
sealed until input degradation and COCO-to-YOLO semantics are independently
verified against training-time validation artifacts.

Usage:
  python scripts/audit_flir_test_integrity_step11j.py --out reports/test/step11j_audit_v1
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image
import yaml

PROTO='STEP11J_FLIR_DISJOINT_TEST_PREPARATION_READ_ONLY_AUDIT_V1'
PARTS={'train':'images_thermal_train','val':'images_thermal_val','test':'video_thermal_test'}
ALLOWED_METHODS=('pil_bicubic','pil_bilinear','pil_lanczos','cv2_area','cv2_cubic','cv2_linear')


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open('rb') as h:
        for part in iter(lambda:h.read(1024*1024),b''):
            digest.update(part)
    return digest.hexdigest()


def safe_rel_filename(record:dict)->Path:
    raw=record.get('file_name','')
    if not isinstance(raw,str) or '\\' in raw:
        raise ValueError('Invalid COCO file_name')
    p=Path(raw)
    if not raw or p.is_absolute() or '..' in p.parts or len(p.parts)<2:
        raise ValueError(f'Unsafe/unsupported COCO image filename: {raw}')
    return p


def video_id(im:dict)->str:
    name=Path(str(im.get('file_name',''))).name
    vid=im.get('extra_info',{}).get('video_id')
    if not vid and '-frame-' in name:
        vid=name.split('-frame-',1)[0].removeprefix('video-')
    if not vid or re.search(r'[/\\\s]',str(vid)):
        raise ValueError(f'No reliable video ID: {name}')
    return str(vid).removeprefix('video-')


def validate_config(cfg:dict):
    assert isinstance(cfg,dict)
    names={int(k):str(v) for k,v in cfg['model_class_names'].items()}
    mapping={int(k):int(v) for k,v in cfg['coco_category_to_yolo'].items()}
    if names!={0:'person',1:'bike',2:'car',3:'motor',4:'bus',5:'truck',6:'other_vehicle',7:'light',8:'sign'}:
        raise ValueError('Detector classes differ from frozen checkpoint')
    if mapping!={1:0,2:1,3:2,4:3,6:4,8:5,79:6,10:7,12:8}:
        raise ValueError('Unexpected COCO mapping; audit training conversion first')
    if sorted(set(mapping.values()))!=list(range(9)):
        raise ValueError('Missing model class')
    if tuple(cfg['hr_size'])!=(640,512) or tuple(cfg['lr_size'])!=(320,256):
        raise ValueError('Unexpected shapes')
    return names,mapping


def coco_bbox_to_yolo(ann:dict,image:dict,mapping:dict[int,int]):
    """Return normalized class+box, ignoring excluded categories / crowd.

    Note: whether the original training converter used precisely these rules is
    an empirical hypothesis, checked against actual val YOLO labels below.
    """
    cls=mapping.get(int(ann['category_id']))
    if cls is None or bool(ann.get('iscrowd',False)):
        return None
    width,height=int(image['width']),int(image['height'])
    x,y,w,h=map(float,ann['bbox'])
    if not all(np.isfinite(q) for q in (x,y,w,h)) or w<=0 or h<=0 or width<=0 or height<=0:
        raise ValueError('Invalid COCO bbox')
    x0=max(0.,min(float(width),x));y0=max(0.,min(float(height),y))
    x1=max(0.,min(float(width),x+w));y1=max(0.,min(float(height),y+h))
    if x1<=x0 or y1<=y0:
        return None
    return (cls,((x0+x1)/2)/width,((y0+y1)/2)/height,(x1-x0)/width,(y1-y0)/height)


def read_yolo(path:Path)->list[tuple[int,float,float,float,float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        v=line.split()
        if len(v)!=5:raise ValueError('Expected five-value YOLO labels')
        cls=float(v[0]); b=list(map(float,v[1:]))
        if not cls.is_integer() or not 0<=cls<=8 or any(not np.isfinite(x) for x in b):
            raise ValueError('Invalid YOLO label')
        result.append((int(cls),*b))
    return result


def match_label_sets(reference:list,constructed:list,tol:float=0.00002):
    """Class-aware order-invariant matching with duplicate multiplicity."""
    if len(reference)!=len(constructed):
        return False,{'reference':len(reference),'constructed':len(constructed)}
    # Deterministic greedy; boxes occasionally duplicate, so track consumption.
    pending=list(constructed)
    for row in reference:
        candidate=next((i for i,p in enumerate(pending)
                        if p[0]==row[0] and max(abs(p[k]-row[k]) for k in range(1,5))<=tol),None)
        if candidate is None:
            return False,{'reference':len(reference),'constructed':len(constructed),
                          'unmatched_label':row}
        pending.pop(candidate)
    return True,{'reference':len(reference),'constructed':len(constructed)}


def low_resolution(image:Image.Image,method:str)->np.ndarray:
    """Candidate preprocessing hypotheses; never assume which is original."""
    if image.size!=(640,512):
        raise ValueError('Thermal source is not 640x512')
    gray=np.asarray(image.convert('L'),dtype=np.uint8)
    if method=='pil_bicubic':
        out=np.asarray(image.convert('L').resize((320,256),Image.Resampling.BICUBIC))
    elif method=='pil_bilinear':
        out=np.asarray(image.convert('L').resize((320,256),Image.Resampling.BILINEAR))
    elif method=='pil_lanczos':
        out=np.asarray(image.convert('L').resize((320,256),Image.Resampling.LANCZOS))
    elif method in ('cv2_area','cv2_cubic','cv2_linear'):
        m={'cv2_area':cv2.INTER_AREA,'cv2_cubic':cv2.INTER_CUBIC,'cv2_linear':cv2.INTER_LINEAR}[method]
        out=cv2.resize(gray,(320,256),interpolation=m)
    else:
        raise ValueError(f'Unknown candidate method {method}')
    if out.dtype!=np.uint8 or out.shape!=(256,320):
        raise AssertionError('Wrong LR candidate')
    return out


def sample_val(images:list[dict],num:int)->list[dict]:
    """Cover as many video groups as possible, then deterministic evenly spread."""
    ordered=sorted(images,key=lambda im:str(im['file_name']))
    first={}
    for im in ordered:
        first.setdefault(video_id(im),im)
    selected=list(first.values())
    seen={im['id'] for im in selected}
    remains=[im for im in ordered if im['id'] not in seen]
    num=max(len(selected),min(len(images),num))
    needed=num-len(selected)
    if needed:
        # Deterministic distributed ranks, no randomness and no selection by image metrics.
        indices=np.linspace(0,len(remains)-1,needed,dtype=int).tolist()
        selected += [remains[i] for i in indices]
    return selected


def load_dataset(coco_path:Path)->tuple[dict,list[dict],list[dict]]:
    raw=json.loads(coco_path.read_text(encoding='utf-8'))
    images=raw['images'];anns=raw['annotations']
    if not images or not isinstance(anns,list):
        raise ValueError('Missing images or annotations')
    ids=[im['id'] for im in images]
    if len(ids)!=len(set(ids)) or any(not isinstance(x,dict) for x in images):
        raise ValueError('Duplicate/invalid COCO image ID')
    names=[safe_rel_filename(im).as_posix() for im in images]
    if len(names)!=len(set(names)):
        raise ValueError('Duplicate COCO image filename')
    image_set=set(ids)
    for ann in anns:
        if ann['image_id'] not in image_set:
            raise ValueError('Annotation references unknown image')
    return raw,images,anns


def inspect_split(folder:Path, expected_images:int, expected_sequences:int):
    _,images,anns=load_dataset(folder/'coco.json')
    seqs={video_id(i) for i in images}
    if len(images)!=expected_images or len(seqs)!=expected_sequences:
        raise ValueError(f'Split differs from prior audited counts: {folder}: {len(images)}, {len(seqs)}')
    for im in images:
        rel=safe_rel_filename(im)
        p=folder/rel
        if not p.is_file():
            raise FileNotFoundError(f'Listed COCO source image missing: {p}')
        if (int(im['width']),int(im['height']))!=(640,512):
            raise ValueError(f'Unexpected COCO dimensions {p}')
    by_class=Counter(int(a['category_id']) for a in anns)
    return images,anns,seqs,by_class


def audit(args):
    if args.out.exists():
        raise FileExistsError(f'New report directory required: {args.out}')
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    names,mapping=validate_config(cfg)
    root=Path(cfg['raw_root']); processed=Path(cfg['processed_validation_root'])
    summaries={}; inventory={}
    for split,folder in PARTS.items():
        path=root/folder
        coco_index=json.loads((path/'coco.json').read_text(encoding='utf-8'))
        all_categories={int(c['id']):c['name'] for c in coco_index['categories']}
        expected_names={1:'person',2:'bike',3:'car',4:'motor',6:'bus',8:'truck',79:'other vehicle',10:'light',12:'sign'}
        for cid,expected_name in expected_names.items():
            if all_categories.get(cid)!=expected_name:
                raise ValueError(f'{split} COCO category {cid} name differs from frozen mapping: {all_categories.get(cid)}')
        imgs,anns,seqs,cats=inspect_split(path,int(cfg['expected_images'][split]),
                                           int(cfg['expected_sequences'][split]))
        inventory[split]=(path,imgs,anns,seqs)
        summaries[split]={
            'images':len(imgs),'sequences':len(seqs),'annotation_records':len(anns),
            'annotated_images':len({ann['image_id'] for ann in anns}),
            'category_counts':dict(sorted(cats.items())),
            'video_ids_sha256':hashlib.sha256('\n'.join(sorted(seqs)).encode()).hexdigest(),
            'coco_sha256':sha256(path/'coco.json'),
        }
    for a,b in (('train','test'),('val','test'),('train','val')):
        shared=inventory[a][3]&inventory[b][3]
        if shared:
            raise RuntimeError(f'Sequence leakage: {a} / {b}: {sorted(shared)[:5]}')
    # Validate metadata rules on TEST without using the test boxes to tune a method.
    test_path,test_images,test_anns,_=inventory['test']
    ids={im['id']:im for im in test_images}
    converted=Counter(); dropped=Counter(); crowds=Counter()
    for ann in test_anns:
        cat=int(ann['category_id'])
        if cat not in mapping:dropped[cat]+=1
        elif bool(ann.get('iscrowd',False)):crowds[cat]+=1
        else:
            row=coco_bbox_to_yolo(ann,ids[ann['image_id']],mapping)
            if row is not None:converted[row[0]]+=1
    # VALIDATION input preprocessing matching: exact bytewise pixels, never RMSE proxy.
    val_path,val_images,val_anns,_=inventory['val']
    chosen=sample_val(val_images,int(cfg['validation_pixel_audit_images']))
    exact=Counter(); total={k:0 for k in ALLOWED_METHODS}
    absolute_pixel_error=Counter(); max_error=Counter()
    examples=[]
    for j,im in enumerate(chosen,1):
        source=val_path/safe_rel_filename(im)
        target=processed/'images'/'val'/(Path(str(im['file_name'])).stem+'.png')
        if not target.is_file():raise FileNotFoundError(f'Processed validation LR PNG missing {target}')
        observed=cv2.imread(str(target),cv2.IMREAD_GRAYSCALE)
        if observed is None or observed.shape!=(256,320):
            raise ValueError(f'Bad processed validation LR PNG: {target}')
        with Image.open(source) as raw:
            candidates={m:low_resolution(raw,m) for m in ALLOWED_METHODS}
        row={'file_name':Path(str(im['file_name'])).name,
             'source_sha256':sha256(source),'processed_lr_sha256':sha256(target),
             'match':{},'max_abs_diff':{}}
        for method,arr in candidates.items():
            delta=np.abs(observed.astype(np.int16)-arr.astype(np.int16))
            same=bool(np.array_equal(observed,arr))
            exact[method]+=int(same);total[method]+=1
            absolute_pixel_error[method]+=int(np.sum(delta))
            max_error[method]=max(max_error[method],int(np.max(delta)))
            row['match'][method]=same
            row['max_abs_diff'][method]=int(np.max(np.abs(observed.astype(np.int16)-arr.astype(np.int16))))
        examples.append(row)
        if j==1 or j%20==0 or j==len(chosen):
            print(f'Validation image preprocessing calibrated {j}/{len(chosen)}',flush=True)
    matches=[k for k in ALLOWED_METHODS if exact[k]==total[k] and total[k]==len(chosen)]
    # VALIDATION label semantics matching for all 1,144 val frames. Fail closed if inconsistent.
    val_by_img=defaultdict(list)
    for ann in val_anns:val_by_img[ann['image_id']].append(ann)
    label_failures=[];label_counts=Counter();skipped_cat=Counter()
    for i,im in enumerate(val_images):
        label=processed/'labels'/'val'/(Path(str(im['file_name'])).stem+'.txt')
        actual=read_yolo(label)
        generated=[]
        for ann in val_by_img[im['id']]:
            if int(ann['category_id']) not in mapping:
                skipped_cat[int(ann['category_id'])]+=1
            result=coco_bbox_to_yolo(ann,im,mapping)
            if result is not None:generated.append(result)
        yes,detail=match_label_sets(actual,generated)
        label_counts['passed' if yes else 'failed']+=1
        if not yes and len(label_failures)<25:
            label_failures.append({'image':Path(im['file_name']).name,**detail})
    # Distinguish empty labels and skipped categories, do not omit empty test frames.
    summary={
        'protocol':PROTO,'created_utc':datetime.now(timezone.utc).isoformat(),
        'config_sha256':sha256(args.config),'dataset_splits':summaries,
        'sequence_disjoint':True,'model_class_names':names,
        'coco_category_to_yolo':mapping,
        'excluded_test_categories':dict(sorted(dropped.items())),
        'excluded_test_crowds':dict(sorted(crowds.items())),
        'included_test_annotation_classes':dict(sorted(converted.items())),
        'test_source': '8-bit COCO-referenced thermal JPG; raw TIFF is NOT used',
        'validation_image_calibration':{
            'frames':len(chosen),'exact_image_matches_by_method':dict(exact),
            'exactly_matching_methods':matches,
            'mean_absolute_pixel_error_by_method':{m:round(absolute_pixel_error[m]/(len(chosen)*320*256),6) for m in ALLOWED_METHODS},
            'max_abs_pixel_error_by_method':dict(max_error),
            'example_rows':examples[:3],
        },
        'validation_label_calibration':{'images':len(val_images),
            'matches':int(label_counts['passed']),'mismatches':int(label_counts['failed']),
            'example_failures':label_failures,'ignored_val_categories':dict(sorted(skipped_cat.items()))},
        'ready_for_test_preparation':bool(matches and not label_counts['failed']),
        'limitations':[
            'Metadata and training-pipeline parity only, NO predictions on test images.',
            'Video IDs are disjoint; cannot rule out cross-sequence near-duplicate image content.',
            'Test JPG is not raw radiometric TIFF; pairing to existing 8-bit model is intentional.',
            'COCO excluded categories are ignored as unlabeled by current nine-class detector; diagnostic FPs may include them.',
            'A test split of FLIR ADAS v2, NOT external cross-dataset generalization.',
            'Preprocessing and label conversion require validation parity before any test evaluator.',
            'Do not use this test split for method/threshold hyperparameter selection.',
        ]}
    args.out.mkdir(parents=True)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('\nSTEP11J — dataset source/parity AUDIT; NO TEST INFERENCE')
    print('Test images:',len(test_images),'test sequences:',len(inventory['test'][3]))
    print('Selected 8-bit JPG source, raw TIFF unmodified')
    print('Val pixel calibration:',dict(exact),'of',len(chosen))
    print('Val label matches:',label_counts['passed'],'/',len(val_images))
    print('Safe to prepare?',summary['ready_for_test_preparation'])
    print('SAVED:',args.out/'summary.json')
    if not summary['ready_for_test_preparation']:
        raise RuntimeError('NOT READY: training-time LR or label conversion parity not established. '
                           'Inspect saved report; do not create test dataset.')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/flir_test_integrity_step11j.yaml'))
    parser.add_argument('--out',required=True,type=Path)
    audit(parser.parse_args())

if __name__=='__main__':main()
