import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_large_fsrcnn_step9h import analyze_pair, read_detections


def fixtures():
    return {
        "m": {
            "a": {"sequence": "s1", "small_tp": 3, "fp": 2, "small_gt": 5},
            "b": {"sequence": "s1", "small_tp": 2, "fp": 1, "small_gt": 5},
            "c": {"sequence": "s2", "small_tp": 1, "fp": 0, "small_gt": 5},
        },
        "r": {
            "a": {"sequence": "s1", "small_tp": 2, "fp": 4, "small_gt": 5},
            "b": {"sequence": "s1", "small_tp": 2, "fp": 1, "small_gt": 5},
            "c": {"sequence": "s2", "small_tp": 2, "fp": 0, "small_gt": 5},
        },
    }


def make_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'sequence', 'method', 'small_tp', 'fp', 'small_gt'])
        writer.writeheader()
        writer.writerows(rows)


def test_correct_delta_and_groups():
    result, rows = analyze_pair(fixtures(), 'm', 'r', 0.25, 1000, 42)
    assert result['delta_small_tp'] == 0
    assert result['delta_fp'] == -2
    assert result['delta_utility_total'] == 0.5
    assert result['delta_utility_per_image'] == pytest.approx(1 / 6)
    assert result['num_sequences'] == 2
    assert result['sequence_wins_ties_losses'] == {'wins': 1, 'ties': 0, 'losses': 1}
    assert len(rows) == 2


def test_bootstrap_deterministic():
    a, _ = analyze_pair(fixtures(), 'm', 'r', 0.25, 1000, 42)
    b, _ = analyze_pair(fixtures(), 'm', 'r', 0.25, 1000, 42)
    assert a['sequence_bootstrap_95ci_per_image'] == b['sequence_bootstrap_95ci_per_image']


def test_expected_bootstrap_extremes():
    a, _ = analyze_pair(fixtures(), 'm', 'r', 0.25, 1000, 42)
    lo, hi = a['sequence_bootstrap_95ci_per_image']
    assert lo >= -1.0 and hi <= 1.0
    assert lo < 0 < hi


def test_missing_method_rejected():
    with pytest.raises(ValueError):
        analyze_pair(fixtures(), 'missing', 'r', 0.25)


def test_negative_penalty_rejected():
    with pytest.raises(ValueError):
        analyze_pair(fixtures(), 'm', 'r', -0.5)


def test_too_few_resamples_rejected():
    with pytest.raises(ValueError):
        analyze_pair(fixtures(), 'm', 'r', 0.25, 50)


def test_csv_round_trip(tmp_path):
    rows = [dict(image=i, method=m, **r) for m, images in fixtures().items() for i, r in images.items()]
    path = tmp_path / 'results.csv'
    make_csv(path, rows)
    assert read_detections(path) == fixtures()


def test_duplicate_rejected(tmp_path):
    row = dict(image='a', method='m', sequence='s', small_tp=1, fp=0, small_gt=1)
    path = tmp_path / 'results.csv'
    make_csv(path, [row, row])
    with pytest.raises(ValueError, match='Duplicate'):
        read_detections(path)


def test_mismatched_images_rejected(tmp_path):
    rows = [
        dict(image='a', method='m', sequence='s', small_tp=1, fp=0, small_gt=1),
        dict(image='b', method='r', sequence='s', small_tp=1, fp=0, small_gt=1),
    ]
    path = tmp_path / 'results.csv'
    make_csv(path, rows)
    with pytest.raises(ValueError, match='Non-identical image set'):
        read_detections(path)


def test_inconsistent_gt_rejected(tmp_path):
    rows = [
        dict(image='a', method='m', sequence='s', small_tp=1, fp=0, small_gt=2),
        dict(image='a', method='r', sequence='s', small_tp=1, fp=0, small_gt=3),
    ]
    path = tmp_path / 'results.csv'
    make_csv(path, rows)
    with pytest.raises(ValueError, match='mismatch'):
        read_detections(path)


def test_tp_exceeds_gt_rejected(tmp_path):
    rows = [dict(image='a', method='m', sequence='s', small_tp=5, fp=0, small_gt=3)]
    path = tmp_path / 'results.csv'
    make_csv(path, rows)
    with pytest.raises(ValueError, match='exceeds'):
        read_detections(path)


def test_negative_counts_rejected(tmp_path):
    rows = [dict(image='a', method='m', sequence='s', small_tp=0, fp=-1, small_gt=1)]
    path = tmp_path / 'results.csv'
    make_csv(path, rows)
    with pytest.raises(ValueError, match='Negative'):
        read_detections(path)
