"""CPU-only tests for UtiTherm-SR Step 5B's data, split, model and ranking."""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from utitherm.utility.predictor import HEADS, UtilityPredictor, encode_tile, outcome_labels, ranking_score
from train_utility_predictor import (UtilityDataset, grouped_split, load_measurements,
                                    ranking_report, weight_by_head)


def sample_rows(num_seqs=8):
    data = []
    for i in range(num_seqs):
        for t in range(16):
            delta_tp = 1 if t == 10 and i % 2 == 0 else (-1 if t == 6 and i % 2 == 1 else 0)
            delta_fp = -1 if t == 5 else (1 if t == 9 else 0)
            score = delta_tp - 0.25 * delta_fp
            row = {"image": f"video-id{i}-frame-000001.png", "sequence": f"video-id{i}",
                   "tile_index": str(t), "tile_x0_hr": str((t % 4) * 160),
                   "tile_y0_hr": str((t // 4) * 128), "tile_x1_hr": str((t % 4 + 1) * 160),
                   "tile_y1_hr": str((t // 4 + 1) * 128), "delta_small_tp": str(delta_tp),
                   "delta_fp": str(delta_fp), "utility_score": str(score),
                   "positive_label": str(int(score > 0))}
            row["rect"] = (int(row["tile_x0_hr"]), int(row["tile_y0_hr"]),
                            int(row["tile_x1_hr"]), int(row["tile_y1_hr"]))
            row["score"] = score
            row["target"] = outcome_labels(row)
            data.append(row)
    return data


def test_head_label_semantics_and_surrogate_ranking():
    row = {"delta_small_tp": "1", "delta_fp": "-2"}
    assert list(outcome_labels(row)) == [1, 1, 0, 0]
    assert list(outcome_labels({"delta_small_tp": "-1", "delta_fp": "2"})) == [0, 0, 1, 1]
    probs = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    assert np.allclose(ranking_score(probs), [1.25, -1.25])


def test_no_ground_truth_used_as_input_and_location_correct():
    img = np.arange(256 * 320, dtype=np.uint32).reshape(256, 320).astype(np.uint8)
    x, loc = encode_tile(img, (0, 0, 160, 128))
    assert x.shape == (2, 64, 80)
    assert np.allclose(loc.numpy(), [.125, .125, .25, .25])
    assert x.min() >= 0 and x.max() <= 1
    assert x[0, 0, 0] == pytest.approx(img[0, 0] / 255, abs=1e-6)
    assert x[1].shape == (64, 80)


def test_tile_coordinate_checks():
    img = np.zeros((256, 320), dtype=np.uint8)
    with pytest.raises(ValueError):
        encode_tile(img, (-1, 0, 160, 128))
    with pytest.raises(ValueError):
        encode_tile(img.astype(np.float32), (0, 0, 160, 128))


def test_predictor_forward_and_backward_cpu():
    model = UtilityPredictor()
    x = torch.rand(2, 2, 64, 80)
    coords = torch.rand(2, 4)
    logits = model(x, coords)
    assert logits.shape == (2, 4)
    assert list(HEADS) == ["small_gain", "fp_reduce", "small_loss", "fp_increase"]
    logits.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
    with pytest.raises(ValueError):
        model(x[:, :1], coords)


def test_video_sequence_split_disjoint_and_deterministic():
    rows = sample_rows(10)
    a, v = grouped_split(rows, .20, 42)
    a2, v2 = grouped_split(rows, .20, 42)
    assert [r["image"] for r in a] == [r["image"] for r in a2]
    assert [r["image"] for r in v] == [r["image"] for r in v2]
    assert len(a) == 128 and len(v) == 32
    assert not ({r["sequence"] for r in a} & {r["sequence"] for r in v})


def test_class_imbalance_cap():
    rows = sample_rows()
    weights = weight_by_head(rows, cap=10)
    assert weights.shape == (4,)
    assert torch.isfinite(weights).all()
    assert weights.min() >= 1 and weights.max() <= 10


def test_rank_budget_vs_oracle():
    rows = sample_rows(5)
    # Perfect classifier heads for sample synthetic rows.
    probs = np.stack([r["target"] for r in rows]).astype(float)
    metrics = ranking_report(rows, probs, budget_tiles=4, seed=42)
    assert metrics["num_images"] == 5
    assert metrics["budget_tiles"] == 4
    assert metrics["mean_marginal_utility_per_image"]["oracle"] >= metrics["mean_marginal_utility_per_image"]["predicted"]
    assert metrics["mean_marginal_utility_per_image"]["predicted"] > metrics["mean_marginal_utility_per_image"]["random"]


def test_csv_integrity_validation(tmp_path):
    rows = sample_rows(4)
    csv_path = tmp_path / "roi_utility.csv"
    cols = [k for k in rows[0] if k not in ("rect", "target", "score")]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: r[k] for k in cols} for r in rows])
    valid = load_measurements(csv_path)
    assert len(valid) == 64
    # Corrupted target score must fail hard.
    rows[0]["utility_score"] = "50"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: r[k] for k in cols} for r in rows])
    with pytest.raises(ValueError, match="utility-label mismatch"):
        load_measurements(csv_path)


def test_dataset_loading_single_grayscale_frame(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    rows = sample_rows(4)
    for name in set(r["image"] for r in rows):
        assert cv2.imwrite(str(root / name), np.zeros((256, 320), dtype=np.uint8))
    ds = UtilityDataset(rows, root, (80, 64))
    x, coords, labels = ds[0]
    assert x.shape == (2, 64, 80)
    assert coords.shape == (4,)
    assert labels.shape == (4,)
