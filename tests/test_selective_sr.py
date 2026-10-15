"""Step 6A unit tests. No CUDA/YOLO or FLIR file downloads needed."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from evaluate_selective_sr import (
    POLICIES, HR_SIZE, get_eval_image_names, choose_tiles, apply_tiles,
    build_policy_image, policy_summary, predict_tile_scores,
)
from probe_roi_utility import roi_grid, enhanced_tile
from utitherm.utility.predictor import encode_tile, HEADS, ranking_score


class PredictStub(nn.Module):
    def forward(self, x, coords):
        # Deterministic logits depending on normalized geometric positions.
        return torch.stack([3 * coords[:, 0], -coords[:, 1],
                            coords[:, 1], coords[:, 0]], dim=1)


def fake_split(tmp_path: Path):
    csv_file = tmp_path / 'utility.csv'
    with csv_file.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['image', 'sequence', 'tile_index'])
        wr.writeheader()
        for seq in ['video-A', 'video-B', 'video-C']:
            for i in range(16):
                wr.writerow({'image': f'{seq}-frame-000001-suffix.png',
                             'sequence': seq, 'tile_index': i})
    meta = tmp_path / 'split.json'
    meta.write_text(json.dumps({
        'train_sequences': ['video-A'], 'val_sequences': ['video-B', 'video-C'],
        'intervention_csv_sha256': hashlib.sha256(csv_file.read_bytes()).hexdigest(),
    }), encoding='utf-8')
    return csv_file, meta


def test_split_uses_predictor_validation_only(tmp_path):
    c, s = fake_split(tmp_path)
    result = get_eval_image_names(c, s, 0)
    assert len(result) == 2
    assert all('video-A' not in p for p in result)
    assert get_eval_image_names(c, s, 1) == result[:1]


def test_rejects_split_hash_mismatch(tmp_path):
    c, s = fake_split(tmp_path)
    with c.open('a', encoding='utf-8') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='SHA256'):
        get_eval_image_names(c, s, 0)


def test_rejects_incomplete_tile_set(tmp_path):
    c, s = fake_split(tmp_path)
    lines = c.read_text().splitlines()
    # Remove final tile in validation sequence and fix SHA.
    c.write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
    meta = json.loads(s.read_text())
    meta['intervention_csv_sha256'] = hashlib.sha256(c.read_bytes()).hexdigest()
    s.write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError, match='16'):
        get_eval_image_names(c, s)


def test_rejects_train_val_overlap(tmp_path):
    c, s = fake_split(tmp_path)
    meta = json.loads(s.read_text())
    meta['train_sequences'].append('video-B')
    s.write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError, match='overlap'):
        get_eval_image_names(c, s)


def test_grid_budget_and_deterministic_random():
    tiles = roi_grid(640, 512, 4, 4)
    assert len(tiles) == 16
    for policy in ('fixed_first4', 'random4'):
        a = choose_tiles(policy, 'frame-x.png', 4, 16, 42)
        b = choose_tiles(policy, 'frame-x.png', 4, 16, 42)
        assert len(a) == 4 and len(set(a)) == 4 and a == b
    assert choose_tiles('fixed_first4', 'frame-x.png', 4, 16, 42) == [0,1,2,3]


def test_predictor_top_tiles_stable_and_no_ground_truth():
    scores = np.array([i for i in range(16)], dtype=np.float32)
    assert choose_tiles('predictor4', 'x.png', 4, 16, 42, scores) == [15,14,13,12]
    with pytest.raises(ValueError):
        choose_tiles('predictor4', 'x.png', 4, 16, 42, scores[:15])
    with pytest.raises(ValueError):
        choose_tiles('predictor4', 'x.png', 17, 16, 42, scores)


def test_predictor_encoder_matches_training_contract():
    gray = np.arange(320*256, dtype=np.uint32).reshape(256,320).astype(np.uint8)
    tiles = roi_grid(640, 512, 4, 4)
    scores = predict_tile_scores(PredictStub(), gray, tiles, (80,64), torch.device('cpu'), 0.25)
    assert scores.shape == (16,) and np.isfinite(scores).all()
    # Explicitly check direct batched forward agrees with the helper.
    enc = [encode_tile(gray, rect) for rect in tiles]
    x = torch.stack([p for p,_ in enc]); coord = torch.stack([p for _,p in enc])
    expected = ranking_score(PredictStub()(x, coord).sigmoid().detach().numpy(), 0.25)
    assert np.allclose(scores, expected)


def test_multi_tile_composition_is_pixel_local():
    base = np.full((512,640), 50, np.uint8)
    cubic = np.full((512,640), 150, np.uint8)
    tiles = roi_grid(640,512,4,4)
    selected = [2,9,14,15]
    altered = apply_tiles(base,cubic,tiles,selected,8)
    diff = altered != base
    mask = np.zeros_like(base, dtype=bool)
    for i in selected:
        x0,y0,x1,y1 = tiles[i]
        mask[y0:y1,x0:x1] = True
    assert not np.any(diff & ~mask)
    assert diff.sum() > 0 and altered[0,0] == 50
    assert np.array_equal(base, np.full((512,640),50,np.uint8))


def test_single_tile_matches_step5a_counterfactual():
    base = np.full((512,640), 50, np.uint8)
    cubic = np.full((512,640), 150, np.uint8)
    tiles = roi_grid(640,512,4,4)
    i = 8
    assert np.array_equal(apply_tiles(base,cubic,tiles,[i],8),
                          enhanced_tile(base,cubic,tiles[i],8))


def test_full_and_selective_compositions_are_different():
    gray = np.tile(np.arange(320, dtype=np.uint16).astype(np.uint8), (256,1))
    tiles = roi_grid(640,512,4,4)
    fixed, chosen, selector_ms = build_policy_image('fixed_first4',gray,tiles,4,42,'x.png',8,None,(80,64),torch.device('cpu'),.25)
    full, chosen_full, _ = build_policy_image('full_bicubic_640',gray,tiles,4,42,'x.png',8,None,(80,64),torch.device('cpu'),.25)
    linear, no_tiles, _ = build_policy_image('linear_640',gray,tiles,4,42,'x.png',8,None,(80,64),torch.device('cpu'),.25)
    assert len(chosen) == 4 and len(chosen_full) == 16 and not no_tiles
    assert fixed.shape == full.shape == linear.shape == (512,640)
    assert np.array_equal(full, cv2.resize(gray,(640,512),interpolation=cv2.INTER_CUBIC))
    # Unselected bottom half must be unchanged vs linear.
    assert np.array_equal(fixed[128:], linear[128:])


def test_actual_multi_region_utility_uses_counts_not_proxy():
    data = []
    examples = {
        'linear_640': [(10,20,5),(11,19,6)],
        'predictor4': [(12,23,4),(10,21,8)],
        'random4': [(10,20,6),(11,19,5)],
    }
    for policy, values in examples.items():
        for i, (tp,alltp,fp) in enumerate(values):
            data.append({'policy':policy,'small_gt':30,'small_tp':tp,'all_tp':alltp,'fp':fp,
                         'prepare_ms':1,'selector_ms':.1,'detect_ms':2,'pipeline_ms':3,
                         'enhanced_fraction':.25 if policy!='linear_640' else 0})
    result = policy_summary(data,.25)
    assert result['predictor4']['delta_small_tp_vs_linear'] == 1
    assert result['predictor4']['delta_fp_vs_linear'] == 1
    assert result['predictor4']['observed_utility_total_vs_linear'] == .75
    assert result['linear_640']['observed_utility_total_vs_linear'] == 0


def test_policy_names_frozen_to_research_controls():
    assert POLICIES == ('linear_640','fixed_first4','random4','predictor4','full_bicubic_640')
