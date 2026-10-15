"""CPU-only deterministic tests for Step 10E data integrity and paired analysis."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_large_sr_step10e.py'
spec = importlib.util.spec_from_file_location('step10e', SCRIPT)
a = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = a
spec.loader.exec_module(a)


@pytest.fixture
def fixture_data(tmp_path):
    detfile, timefile, summfile = (tmp_path / name for name in ('det.csv', 'timing.csv', 'source.json'))
    imginfo = [('f1.png', 'S1', 3), ('f2.png', 'S1', 2),
               ('f3.png', 'S2', 4), ('f4.png', 'S2', 2)]
    counts = {
        'v2': [(1, 2), (1, 2), (2, 3), (1, 2)],
        'v3': [(2, 3), (1, 1), (3, 4), (1, 3)],
        'bic': [(2, 2), (2, 2), (3, 2), (2, 2)],
    }
    drows = []
    trows = []
    for i, (img, seq, gt) in enumerate(imginfo):
        for method, n in counts.items():
            tp, fp = n[i]
            drows.append({'image': img, 'sequence': seq, 'method': method,
                          'small_tp': tp, 'all_tp': tp + 1, 'fp': fp, 'small_gt': gt})
            for rep in range(3):
                trows.append({'image': img, 'sequence': seq, 'repeat': rep,
                              'method': method, 'selector_ms': 1.0 if method != 'bic' else 0,
                              'sr_ms': 3.0 if method != 'bic' else 0.2,
                              'detector_ms': 10.0, 'total_ms': (14.0 if method != 'bic' else 10.2) + rep * 0.1,
                              'computed_pixels': 33600 if method != 'bic' else 0,
                              'peak_torch_mib': 70})
    with detfile.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, a.DETECTION_FIELDS)
        writer.writeheader(); writer.writerows(drows)
    with timefile.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, a.TIMING_FIELDS)
        writer.writeheader(); writer.writerows(trows)
    source_methods = {}
    for method in counts:
        rows = [r for r in drows if r['method'] == method]
        source_methods[method] = {k: sum(r[k] for r in rows)
                                  for k in ('small_tp', 'all_tp', 'fp', 'small_gt')}
        source_methods[method]['mean_of_image_median_total_ms'] = 14.1 if method != 'bic' else 10.3
    source = {'protocol': 'FLIR_VALIDATION_DEVELOPMENT_NOT_FINAL_TEST',
              'frozen_artifact_checks_verified': True,
              'results': {'images': 4, 'unique_sequences': 2,
                          'repeats_per_image': 3, 'methods': source_methods}}
    summfile.write_text(json.dumps(source))
    cfg = {'detection_csv': str(detfile), 'timing_csv': str(timefile),
           'source_summary': str(summfile), 'output_dir': str(tmp_path / 'outputs'),
           'expected_images': 4, 'expected_sequences': 2, 'expected_repeats': 3,
           'fp_penalty': .25, 'fp_penalty_sensitivity': [0, .25, .5],
           'bootstrap_repetitions': 250, 'bootstrap_seed': 76,
           'worst_images_per_pair': 2, 'pairs': [['v3', 'v2'], ['v3', 'bic']]}
    return cfg, drows, trows


def test_csv_missing_columns(tmp_path):
    bad = tmp_path / 'bad.csv'
    bad.write_text('image,sequence\na,b\n')
    with pytest.raises(ValueError, match='missing columns'):
        a.read_csv(bad, a.DETECTION_FIELDS)


def test_csv_empty(tmp_path):
    p = tmp_path / 'empty.csv'; p.write_text(','.join(a.DETECTION_FIELDS) + '\n')
    with pytest.raises(ValueError, match='empty CSV'):
        a.read_csv(p, a.DETECTION_FIELDS)


@pytest.mark.parametrize('value', ['-1', '1.5', 'oops', ''])
def test_invalid_count(value):
    with pytest.raises(ValueError):
        a.as_nonnegative_int(value, 'count')


@pytest.mark.parametrize('value', ['nan', 'inf', '-3', 'oops'])
def test_invalid_timing(value):
    with pytest.raises(ValueError):
        a.as_finite_nonnegative_float(value, 'latency')


def test_load_detections(fixture_data):
    cfg, _, _ = fixture_data
    det, images, methods, nseq = a.load_detections(Path(cfg['detection_csv']), 4, 2)
    assert (len(images), len(methods), nseq) == (4, 3, 2)
    assert det[('f1.png', 'v3')]['small_tp'] == 2


def test_detection_count_guard(fixture_data):
    cfg, _, _ = fixture_data
    with pytest.raises(ValueError, match='Expected 200 images'):
        a.load_detections(Path(cfg['detection_csv']), 200, 2)


def test_detection_duplicate_guard(fixture_data):
    cfg, rows, _ = fixture_data
    p = Path(cfg['detection_csv'])
    with p.open('a', newline='') as f:
        w = csv.DictWriter(f, a.DETECTION_FIELDS); w.writerow(rows[0])
    with pytest.raises(ValueError, match='Duplicate'):
        a.load_detections(p)


def test_detection_gt_consistency_guard(fixture_data):
    cfg, rows, _ = fixture_data
    rows[0]['small_gt'] = 200
    p = Path(cfg['detection_csv'])
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, a.DETECTION_FIELDS); w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError, match='GT or sequence mismatch'):
        a.load_detections(p)


def test_detection_missing_method_guard(fixture_data):
    cfg, rows, _ = fixture_data
    rows.pop()
    p = Path(cfg['detection_csv'])
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, a.DETECTION_FIELDS); w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError, match='Method records missing'):
        a.load_detections(p)


def test_timing_medians(fixture_data):
    cfg, _, _ = fixture_data
    d, images, methods, _ = a.load_detections(Path(cfg['detection_csv']))
    t = a.load_timings(Path(cfg['timing_csv']), images, methods, d, 3)
    assert t[('f2.png', 'v3')]['total_ms'] == pytest.approx(14.1)
    assert t[('f4.png', 'bic')]['total_ms'] == pytest.approx(10.3)


def test_duplicate_timing_guard(fixture_data):
    cfg, _, trows = fixture_data
    p = Path(cfg['timing_csv'])
    with p.open('a', newline='') as f:
        w = csv.DictWriter(f, a.TIMING_FIELDS); w.writerow(trows[0])
    d, images, methods, _ = a.load_detections(Path(cfg['detection_csv']))
    with pytest.raises(ValueError, match='Duplicate timing'):
        a.load_timings(p, images, methods, d, 3)


def test_missing_timing_guard(fixture_data):
    cfg, _, trows = fixture_data
    trows.pop()
    p = Path(cfg['timing_csv'])
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, a.TIMING_FIELDS); w.writeheader(); w.writerows(trows)
    d, images, methods, _ = a.load_detections(Path(cfg['detection_csv']))
    with pytest.raises(ValueError, match='expected 3 trials'):
        a.load_timings(p, images, methods, d, 3)


def test_percentile():
    assert a.percentile([0, 10, 20], 0.75) == 15
    assert a.percentile([0, 10], .5) == 5


def test_percentile_input_guard():
    with pytest.raises(ValueError):
        a.percentile([], .5)


def test_grouped_bootstrap_reproducible():
    rows = [{'sequence': 'A', 'delta_utility': 1.0} for _ in range(10)] + [
        {'sequence': 'B', 'delta_utility': -1.0}]
    assert a.grouped_bootstrap_ci(rows, 7, 500) == a.grouped_bootstrap_ci(rows, 7, 500)


def test_bootstrap_few_groups_guard():
    with pytest.raises(ValueError, match='two sequence'):
        a.grouped_bootstrap_ci([{'sequence':'A', 'delta_utility': 2}], 2, 100)


def test_per_image_differences(fixture_data):
    cfg, _, _ = fixture_data
    d, images, methods, _ = a.load_detections(Path(cfg['detection_csv']))
    t = a.load_timings(Path(cfg['timing_csv']), images, methods, d, 3)
    rows = a.pairwise_rows(d, t, images, 'v3', 'v2', .25)
    assert len(rows) == 4
    assert sum(r['delta_small_tp'] for r in rows) == 2
    assert sum(r['delta_fp'] for r in rows) == 2
    assert sum(r['delta_utility'] for r in rows) == pytest.approx(1.5)
    assert rows[0]['tradeoff'] == 'more_tp_more_fp'


def test_pair_sensitivity(fixture_data):
    cfg, _, _ = fixture_data
    d, images, methods, _ = a.load_detections(Path(cfg['detection_csv']))
    t = a.load_timings(Path(cfg['timing_csv']), images, methods, d, 3)
    rows = a.pairwise_rows(d, t, images, 'v3', 'v2', .25)
    stats, sequences = a.analyze_pair(rows, 'v3', 'v2', [0, .25, .5], 200, 9)
    assert len(sequences) == 2
    assert stats['delta_utility_total'] == pytest.approx(1.5)
    assert stats['utility_fp_penalty_sensitivity'][-1]['delta_total_utility'] == pytest.approx(1.0)
    assert stats['num_sequences'] == 2


def test_full_audit(fixture_data):
    cfg, _, _ = fixture_data
    result = a.audit(cfg)
    assert len(result['methods']) == 3
    assert len(result['comparisons']) == 2
    assert result['comparisons'][0]['delta_utility_total'] == pytest.approx(1.5)
    out = Path(cfg['output_dir'])
    for name in ('summary.json', 'paired_per_image.csv', 'paired_per_sequence.csv',
                 'worst_images.csv', 'timing_by_method.csv'):
        assert (out / name).is_file(), name
    assert result['source_files'][cfg['detection_csv']] == a.sha256(Path(cfg['detection_csv']))


def test_summary_provenance_guard(fixture_data):
    cfg, _, _ = fixture_data
    p = Path(cfg['source_summary'])
    obj = json.loads(p.read_text()); obj['frozen_artifact_checks_verified'] = False
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='frozen-artifact'):
        a.audit(cfg)


def test_summary_mismatch_guard(fixture_data):
    cfg, _, _ = fixture_data
    p = Path(cfg['source_summary'])
    obj = json.loads(p.read_text()); obj['results']['methods']['v3']['fp'] += 1
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='differs from CSV'):
        a.audit(cfg)


def test_source_not_modified(fixture_data):
    cfg, _, _ = fixture_data
    paths = [Path(cfg[x]) for x in ('source_summary', 'detection_csv', 'timing_csv')]
    hashes = [a.sha256(p) for p in paths]
    a.audit(cfg)
    assert hashes == [a.sha256(p) for p in paths]


def test_invalid_penalty_guard(fixture_data):
    cfg, _, _ = fixture_data
    cfg['fp_penalty'] = -1
    with pytest.raises(ValueError, match='FP penalties'):
        a.audit(cfg)


def test_unknown_pair_guard(fixture_data):
    cfg, _, _ = fixture_data
    cfg['pairs'] = [['v3', 'does_not_exist']]
    with pytest.raises(ValueError, match='Unknown pair'):
        a.audit(cfg)
