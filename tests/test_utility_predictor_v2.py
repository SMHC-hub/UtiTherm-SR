"""Step 7B2B safety tests. No YOLO, GPU or FLIR download needed."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from utitherm.utility.predictor import UtilityPredictor, outcome_labels
from train_utility_predictor_v2 import (
    ImageTilesDataset, head_metrics, load_v1, pairwise_ranking_loss, run_epoch,
    sequence_bootstrap_delta, split_v2, topk_per_image, torch_rank_score,
)


def fake_rows(prefix: str, nseq: int) -> list[dict]:
    all_rows = []
    for i in range(nseq):
        seq = f'video-{prefix}{i:03d}'
        img = seq + '-frame-000001.png'
        for t in range(16):
            gain = (t == 9 and i % 2 == 0)
            loss = (t == 5 and i % 2 == 1)
            fp = -1 if t == 2 else (1 if t == 3 else 0)
            gt = int(gain) - int(loss)
            all_rows.append({'image':img,'sequence':seq,'tile_index':str(t),
                'score': gt - .25*fp, 'target':outcome_labels({'delta_small_tp':gt,'delta_fp':fp}),
                'rect':((t%4)*160,(t//4)*128,(t%4+1)*160,(t//4+1)*128)})
    return all_rows


def test_split_no_overlap_and_all_rows():
    old, new = fake_rows('old', 10), fake_rows('new', 33)
    a, b, c, groups = split_v2(old,new,143,.15,.15)
    assert len(a)+len(b)+len(c) == len(old)+len(new)
    assert len(groups['val']) == 5 and len(groups['test']) == 5
    assert len(groups['train']) == 33
    assert not (set(groups['train']) & set(groups['val']))
    assert not (set(groups['train']) & set(groups['test']))
    assert not (set(groups['val']) & set(groups['test']))
    assert {r['sequence'] for r in old} <= set(groups['train'])


def test_split_is_reproducible():
    old,new=fake_rows('old',10),fake_rows('new',33)
    _,_,_,g1=split_v2(old,new,143,.15,.15)
    _,_,_,g2=split_v2(old,new,143,.15,.15)
    assert g1==g2


def test_split_rejects_overlap():
    old = fake_rows('x',10)
    new = fake_rows('x',11)
    with pytest.raises(ValueError, match='overlap'):
        split_v2(old,new,143,.15,.15)


def test_split_rejects_too_few_sequences():
    with pytest.raises(ValueError,match='>=10'):
        split_v2(fake_rows('old',10),fake_rows('new',9),143,.15,.15)


def test_split_rejects_invalid_fractions():
    with pytest.raises(ValueError,match='Fractions'):
        split_v2(fake_rows('old',10),fake_rows('new',33),143,.45,.15)


def test_pairwise_ranking_loss_gradients():
    logits = torch.zeros((2,16,4), requires_grad=True)
    target = torch.zeros(2,16)
    target[:,0] = 1.
    target[:,1] = -1.
    loss = pairwise_ranking_loss(logits,target,.25,4.)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_pairwise_loss_decreases_with_good_ranking():
    target=torch.zeros(1,16)
    target[0,0]=1.
    target[0,1]=-1.
    neutral=torch.zeros(1,16,4)
    good=neutral.clone()
    good[0,0,0]=7.
    good[0,1,2]=7.
    assert pairwise_ranking_loss(good,target,.25,4.) < pairwise_ranking_loss(neutral,target,.25,4.)


def test_pairwise_loss_ties_zero():
    logits = torch.randn(1,16,4,requires_grad=True)
    y = torch.zeros(1,16)
    loss = pairwise_ranking_loss(logits,y,.25,4.)
    assert loss.item()==0.
    loss.backward()
    assert logits.grad is not None


def test_ranking_shape_rejected():
    with pytest.raises(ValueError,match='Expected'):
        pairwise_ranking_loss(torch.zeros(1,4,4),torch.zeros(1,4),.25,4)


def test_torch_score_matches_numpy():
    logits=torch.tensor([[[0.,1.,-1.,2.]]])
    p=logits.sigmoid().numpy()
    expected=p[...,0]-p[...,2]+.25*(p[...,1]-p[...,3])
    assert np.allclose(torch_rank_score(logits,.25).numpy(),expected)


def test_dataset_image_grouping(tmp_path):
    rows=fake_rows('x',2)
    for r in rows[::16]:
        cv2.imwrite(str(tmp_path/r['image']), np.full((256,320), 72,np.uint8))
    ds=ImageTilesDataset(rows,tmp_path,(80,64))
    assert len(ds)==2
    x,c,y,u=ds[0]
    assert x.shape==(16,2,64,80)
    assert c.shape==(16,4)
    assert y.shape==(16,4)
    assert u.shape==(16,)
    assert ds.flat_rows()[0]['tile_index']=='0'


def test_dataset_missing_roi_fails(tmp_path):
    rows=fake_rows('x',1)[:-1]
    cv2.imwrite(str(tmp_path/rows[0]['image']),np.zeros((256,320),np.uint8))
    with pytest.raises(ValueError,match='Incomplete'):
        ImageTilesDataset(rows,tmp_path,(80,64))


def test_topk_selects_highest_measured_utility():
    rows=fake_rows('x',2)
    probs=np.zeros((len(rows),4))
    for i,r in enumerate(rows):
        if r['score'] > 0:
            probs[i,0]=1
        if r['score'] < 0:
            probs[i,2]=1
    selections=topk_per_image(rows,probs,4,.25)
    assert len(selections)==2
    for item in selections.values():
        assert len(item['tiles'])==4
        assert item['utility'] >= 0


def test_cluster_bootstrap_paired():
    a={f'im{i}':{'sequence':f's{i//2}','utility':1.} for i in range(8)}
    b={k:{'sequence':v['sequence'],'utility':2.} for k,v in a.items()}
    out=sequence_bootstrap_delta(a,b,143,50)
    assert out['delta_v2_minus_v1_per_sequence_mean']==1.
    assert out['sequence_bootstrap_exploratory_95ci']==[1.,1.]
    assert out['num_sequences']==4


def test_cluster_bootstrap_rejects_invalid():
    with pytest.raises(ValueError):
        sequence_bootstrap_delta({'a':{'sequence':'s1','utility':0}}, {},143)
    with pytest.raises(ValueError,match='At least 2'):
        sequence_bootstrap_delta({'a':{'sequence':'s1','utility':0}},
                                 {'a':{'sequence':'s1','utility':1}},143)


def test_head_metrics_handles_rare_classes():
    true=np.zeros((16,4))
    pred=np.full((16,4),.1)
    true[0,0]=1
    met=head_metrics(true,pred)
    assert met['small_gain']['positive_count']==1
    assert met['small_gain']['average_precision'] is not None
    assert met['small_loss']['average_precision'] is None


def test_old_checkpoint_forward_compatible(tmp_path):
    path=tmp_path/'v1.pt'
    model=UtilityPredictor()
    torch.save({'heads':['small_gain','fp_reduce','small_loss','fp_increase'],
                'input_size':[80,64], 'model':model.state_dict()}, path)
    loaded=load_v1(path,torch.device('cpu'),(80,64))
    with torch.no_grad():
        assert loaded(torch.zeros(2,2,64,80),torch.zeros(2,4)).shape==(2,4)
    with pytest.raises(ValueError,match='incompatible'):
        load_v1(path,torch.device('cpu'),(32,32))


def test_train_step_works_cpu(tmp_path):
    rows=fake_rows('x',2)
    for r in rows[::16]:
        cv2.imwrite(str(tmp_path/r['image']),np.full((256,320),72,np.uint8))
    loader=DataLoader(ImageTilesDataset(rows,tmp_path,(80,64)),batch_size=1)
    model=UtilityPredictor()
    optim=torch.optim.Adam(model.parameters(), lr=1e-4)
    loss=nn.BCEWithLogitsLoss()
    result,_,_=run_epoch(model,loader,torch.device('cpu'),loss,.25,.6,4.,optim)
    assert np.isfinite(result['total'])
    assert result['ranking']>=0
