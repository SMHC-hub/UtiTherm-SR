"""CPU-only scientific contract tests for Step 5A ROI utility target generator."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_roi_utility import roi_grid, enhanced_tile, diverse_subset, utility, detection_counts


def test_grid_covers_every_pixel_once():
    tiles = roi_grid(640, 512, 4, 4)
    assert len(tiles) == 16
    mask = np.zeros((512, 640), np.int32)
    for x0, y0, x1, y1 in tiles:
        mask[y0:y1, x0:x1] += 1
    assert np.all(mask == 1)


def test_tile_modification_is_local():
    linear = np.full((18, 20), 100, np.uint8)
    cubic = np.full((18, 20), 200, np.uint8)
    out = enhanced_tile(linear, cubic, (5, 4, 15, 14), feather=2)
    assert out[0, 0] == 100
    assert out[9, 10] == 200
    assert np.array_equal(out[:4, :], linear[:4, :])
    assert np.array_equal(out[14:, :], linear[14:, :])
    assert np.array_equal(linear, np.full((18, 20), 100, np.uint8))


def test_zero_feather_changes_exact_rectangle():
    l = np.zeros((8, 10), np.uint8)
    c = np.full_like(l, 10)
    out = enhanced_tile(l, c, (1, 2, 9, 7), feather=0)
    assert out.sum() == 8 * 5 * 10
    assert out[2:7, 1:9].min() == 10


def test_deterministic_video_diverse_sampling():
    paths = [Path(f"video-A-frame-{i:06}.png") for i in range(10)]
    paths += [Path(f"video-B-frame-{i:06}.png") for i in range(10)]
    a = diverse_subset(paths, 2, 42)
    b = diverse_subset(paths, 2, 42)
    assert a == b
    assert len({p.stem.split("-frame-")[0] for p in a}) == 2
    assert len(diverse_subset(paths, 0, 42)) == 20


def test_utility_rewards_small_object_gains_and_penalizes_fp():
    baseline = {"small_tp": 3, "all_tp": 4, "fp": 8}
    candidate = {"small_tp": 4, "all_tp": 5, "fp": 10}
    u = utility(baseline, candidate, fp_penalty=0.25)
    assert u == {"delta_small_tp": 1, "delta_all_tp": 1, "delta_fp": 2,
                 "utility_score": 0.5, "positive_label": 1}
    assert utility(baseline, {"small_tp": 3, "all_tp": 4, "fp": 12}, 0.25)["positive_label"] == 0


def test_detection_counts_class_aware_matching():
    gts = [{"cls": 0, "box": (0., 0., 6., 6.), "size": "small"}]
    preds = [{"cls": 0, "box": (0., 0., 6., 6.), "conf": 0.9},
             {"cls": 1, "box": (0., 0., 6., 6.), "conf": 0.9}]
    assert detection_counts(gts, preds, 0.25) == {"small_tp": 1, "all_tp": 1, "fp": 1, "small_gt": 1}


def test_reject_bad_grid_and_input_types():
    with pytest.raises(ValueError):
        roi_grid(10, 10, 12, 3)
    with pytest.raises(ValueError):
        enhanced_tile(np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32), (0, 0, 4, 4))
