"""Fast, GPU-free audit tests for Step 7B2A new utility label dataset."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_utility_v2_manifest import (PROTOCOL, audited_candidates, select_round_robin,
                                          sequence_id, sha256, v1_sequences, verify_manifest)
from collect_utility_v2_labels import COLUMNS, shard_is_valid


def make_sample(tmp_path):
    train = tmp_path/'data/processed/yolo_flir_lr_x2/images/train'
    val = tmp_path/'data/processed/yolo_flir_lr_x2/images/val'
    labels = train.parent.parent/'labels/train'
    for folder in (train, val, labels):
        folder.mkdir(parents=True)
    v1a = 'video-v1old0'
    v1b = 'video-v1old1'
    for seq in [v1a, v1b]+[f'video-fresh{i}' for i in range(8)]+['video-valseq']:
        for j in range(4):
            name = f'{seq}-frame-{j:06}-test.png'
            (train/name).write_bytes(name.encode())
            (labels/f'{Path(name).stem}.txt').write_text('0 0.5 0.5 0.1 0.1\n')
    (val/'video-valseq-frame-000000-val.png').write_bytes(b'v')
    csv_path = tmp_path/'v1.csv'
    csv_path.write_text('sequence\nvideo-v1old0\nvideo-v1old1\n', encoding='utf-8')
    split = tmp_path/'v1_split.json'
    split.write_text(json.dumps({'train_sequences':[v1a], 'val_sequences':[v1b],
                                 'intervention_csv_sha256':sha256(csv_path)}))
    checkpoint = tmp_path/'detector.pt'
    checkpoint.write_bytes(b'detector-weights')
    cfg = dict(train_lr=str(train),val_lr=str(val),v1_roi_csv=str(csv_path),
               v1_split_manifest=str(split), detector_checkpoint=str(checkpoint))
    return cfg


def frozen(cfg, names):
    train=Path(cfg['train_lr'])
    labels=train.parent.parent/'labels/train'
    return dict(protocol=PROTOCOL, train_lr=str(train.resolve()), num_images=len(names),
                unique_sequences=len({sequence_id(x) for x in names}),
                max_per_sequence=3, images=names,
                image_sha256={x:sha256(train/x) for x in names},
                label_sha256={x:sha256(labels/(Path(x).stem+'.txt')) for x in names},
                v1_csv_sha256=sha256(Path(cfg['v1_roi_csv'])),
                v1_split_sha256=sha256(Path(cfg['v1_split_manifest'])),
                detector_checkpoint_sha256=sha256(Path(cfg['detector_checkpoint'])))


def test_sequence_parser_accepts_expected_format():
    assert sequence_id('video-abc-frame-000123-test.png')=='video-abc'


@pytest.mark.parametrize('bad', ['../video-a-frame-1.png',r'..\video-a-frame-1.png',
                                  'video-a-frame-1.jpg', 'img.png'])
def test_sequence_parser_rejects_bad_paths(bad):
    with pytest.raises(ValueError):
        sequence_id(bad)


def test_v1_hash_and_sequence_audit(tmp_path):
    cfg=make_sample(tmp_path)
    assert v1_sequences(Path(cfg['v1_roi_csv']),Path(cfg['v1_split_manifest']))=={'video-v1old0','video-v1old1'}


def test_v1_hash_mismatch_rejected(tmp_path):
    cfg=make_sample(tmp_path)
    Path(cfg['v1_roi_csv']).write_text('sequence\nvideo-corrupt\n')
    with pytest.raises(ValueError,match='hash'):
        v1_sequences(Path(cfg['v1_roi_csv']),Path(cfg['v1_split_manifest']))


def test_candidate_excludes_old_and_validation_sequences(tmp_path):
    cfg=make_sample(tmp_path)
    paths,audit=audited_candidates(Path(cfg['train_lr']),Path(cfg['val_lr']),
                                    v1_sequences(Path(cfg['v1_roi_csv']),Path(cfg['v1_split_manifest'])))
    assert audit['eligible_images']==32
    assert audit['eligible_sequences']==8
    assert audit['v1_sequence_frames_excluded']==8
    assert audit['validation_sequence_frames_excluded']==4
    assert all('fresh' in p.name for p in paths)


def test_select_round_robin_diverse(tmp_path):
    cfg=make_sample(tmp_path)
    paths,_=audited_candidates(Path(cfg['train_lr']),Path(cfg['val_lr']),
                                v1_sequences(Path(cfg['v1_roi_csv']),Path(cfg['v1_split_manifest'])))
    names=select_round_robin(paths,10,142,3)
    assert len(names)==10 and len({sequence_id(p.name) for p in names})==8
    assert max([sum(sequence_id(p.name)==s for p in names) for s in {sequence_id(p.name) for p in names}])<=3
    assert [p.name for p in names]==[p.name for p in select_round_robin(paths,10,142,3)]


def test_insufficient_candidates_returns_less_not_repeat(tmp_path):
    cfg=make_sample(tmp_path)
    paths,_=audited_candidates(Path(cfg['train_lr']),Path(cfg['val_lr']),
                                v1_sequences(Path(cfg['v1_roi_csv']),Path(cfg['v1_split_manifest'])))
    assert len(select_round_robin(paths,50,1,2))==16


def test_verify_manifest_success(tmp_path):
    cfg=make_sample(tmp_path)
    paths,_=audited_candidates(Path(cfg['train_lr']),Path(cfg['val_lr']),
                                v1_sequences(Path(cfg['v1_roi_csv']),Path(cfg['v1_split_manifest'])))
    names=[p.name for p in select_round_robin(paths,10,142,3)]
    assert len(verify_manifest(frozen(cfg,names),cfg))==10


def test_verify_rejects_duplicate_paths(tmp_path):
    cfg=make_sample(tmp_path)
    paths,_=audited_candidates(Path(cfg['train_lr']),Path(cfg['val_lr']),
                                v1_sequences(Path(cfg['v1_roi_csv']),Path(cfg['v1_split_manifest'])))
    n=paths[0].name
    with pytest.raises(ValueError,match='duplicate'):
        verify_manifest(frozen(cfg,[n,n]),cfg)


def test_verify_rejects_changed_label(tmp_path):
    cfg=make_sample(tmp_path)
    paths,_=audited_candidates(Path(cfg['train_lr']),Path(cfg['val_lr']),
                                v1_sequences(Path(cfg['v1_roi_csv']),Path(cfg['v1_split_manifest'])))
    n=paths[0].name
    m=frozen(cfg,[n]); img=Path(cfg['train_lr'])/n
    lab=img.parent.parent.parent/'labels/train'/f'{img.stem}.txt'
    lab.write_text('altered')
    with pytest.raises(ValueError,match='contents changed'):
        verify_manifest(m,cfg)


def test_verify_rejects_v1_collision(tmp_path):
    cfg=make_sample(tmp_path)
    names=['video-v1old0-frame-000000-test.png']
    with pytest.raises(ValueError,match='overlap'):
        verify_manifest(frozen(cfg,names),cfg)


def test_verify_rejects_val_collision(tmp_path):
    cfg=make_sample(tmp_path)
    names=['video-valseq-frame-000000-test.png']
    with pytest.raises(ValueError,match='overlap'):
        verify_manifest(frozen(cfg,names),cfg)


def test_verify_rejects_too_many_from_one_sequence(tmp_path):
    cfg=make_sample(tmp_path)
    names=[f'video-fresh0-frame-{i:06}-test.png' for i in range(4)]
    with pytest.raises(ValueError,match='cap'):
        verify_manifest(frozen(cfg,names),cfg)


def test_shard_validity():
    name='video-test-frame-000001-xyz.png'
    rows=[{'image':name,'sequence':'video-test','tile_index':i,**{k:0 for k in COLUMNS if k not in
           ('image','sequence','tile_index')}} for i in range(16)]
    assert shard_is_valid({'image':name,'rows':rows},name,16)
    assert not shard_is_valid({'image':name,'rows':rows[:-1]},name,16)
    assert not shard_is_valid({'image':name,'rows':rows[:-1]+[rows[0]]},name,16)
    assert not shard_is_valid({'image':name,'rows':rows},'video-other-frame-000001-xyz.png',16)
