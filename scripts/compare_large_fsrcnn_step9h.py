"""Step 9H exploratory paired sequence-cluster analysis for frozen Step 9G development CSV.

Uses paired image-level diagnostic TP/FP, not official mAP or independent final testing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REQUIRED = {"image", "sequence", "method", "small_tp", "fp", "small_gt"}


def read_detections(source: Path) -> dict[str, dict[str, dict]]:
    """Index detection records with strict uniqueness and cross-method GT audits."""
    results: dict[str, dict[str, dict]] = defaultdict(dict)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or REQUIRED - set(reader.fieldnames):
            raise ValueError(f"Missing required CSV columns: {sorted(REQUIRED - set(reader.fieldnames or []))}")
        for row in reader:
            name, method, seq = row["image"], row["method"], row["sequence"]
            if not name or not seq or not method:
                raise ValueError("Blank image, sequence, or method")
            if name in results[method]:
                raise ValueError(f"Duplicate method/image: {method} / {name}")
            record = {
                "sequence": seq,
                "small_tp": int(row["small_tp"]),
                "fp": int(row["fp"]),
                "small_gt": int(row["small_gt"]),
            }
            if min(record["small_tp"], record["fp"], record["small_gt"]) < 0:
                raise ValueError(f"Negative count: {method}/{name}")
            if record["small_tp"] > record["small_gt"]:
                raise ValueError(f"Small TP exceeds small GT: {method}/{name}")
            results[method][name] = record
    if not results:
        raise ValueError("Empty results")
    methods = list(results)
    expected = set(results[methods[0]])
    for method in methods:
        if set(results[method]) != expected:
            raise ValueError(f"Non-identical image set for {method}")
    for image in expected:
        a = results[methods[0]][image]
        if any((results[m][image]["sequence"], results[m][image]["small_gt"]) !=
               (a["sequence"], a["small_gt"]) for m in methods[1:]):
            raise ValueError(f"Sequence or small-GT mismatch for {image}")
    return dict(results)


def analyze_pair(data: dict, method: str, reference: str, penalty: float,
                 bootstrap: int = 10000, seed: int = 42) -> tuple[dict, list[dict]]:
    """Paired difference and 95% sequence-cluster bootstrap CI, per image.

    Resample whole video sequences (not frames) with replacement.  Each draw
    returns image-weighted utility: sum(delta utility)/sum(frames in draw).
    """
    if penalty < 0 or bootstrap < 100 or method not in data or reference not in data:
        raise ValueError("Invalid comparison or bootstrap arguments")
    groups: dict[str, dict] = defaultdict(lambda: {"images": 0, "delta_tp": 0, "delta_fp": 0})
    for image, a in data[method].items():
        b = data[reference][image]
        group = groups[a["sequence"]]
        group["images"] += 1
        group["delta_tp"] += a["small_tp"] - b["small_tp"]
        group["delta_fp"] += a["fp"] - b["fp"]
    sequences = sorted(groups)
    seq_rows: list[dict] = []
    for seq in sequences:
        row = {"method": method, "reference": reference, "sequence": seq, **groups[seq]}
        row["delta_utility"] = row["delta_tp"] - penalty * row["delta_fp"]
        row["delta_utility_per_image"] = row["delta_utility"] / row["images"]
        seq_rows.append(row)
    tp = sum(x["delta_tp"] for x in seq_rows)
    fp = sum(x["delta_fp"] for x in seq_rows)
    util = sum(x["delta_utility"] for x in seq_rows)
    n = sum(x["images"] for x in seq_rows)
    sums = np.array([x["delta_utility"] for x in seq_rows], dtype=np.float64)
    sizes = np.array([x["images"] for x in seq_rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(sequences), size=(bootstrap, len(sequences)))
    samples = sums[choices].sum(axis=1) / sizes[choices].sum(axis=1)
    lo, hi = (float(v) for v in np.quantile(samples, [0.025, 0.975]))
    result = {
        "method": method, "reference": reference,
        "num_images": n, "num_sequences": len(sequences),
        "delta_small_tp": tp, "delta_fp": fp,
        "delta_utility_total": round(util, 6),
        "delta_utility_per_image": round(util / n, 8),
        "sequence_bootstrap_95ci_per_image": [round(lo, 8), round(hi, 8)],
        "sequence_wins_ties_losses": {
            "wins": sum(x["delta_utility"] > 0 for x in seq_rows),
            "ties": sum(x["delta_utility"] == 0 for x in seq_rows),
            "losses": sum(x["delta_utility"] < 0 for x in seq_rows),
        },
    }
    return result, seq_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/large_fsrcnn_utility_step9h.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output directory already exists: {args.out}")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = Path(cfg["detection_csv"])
    data = read_detections(source)
    methods = sorted(data)
    expected_images = int(cfg["expected_images"])
    expected_sequences = int(cfg["expected_sequences"])
    actual_images = len(next(iter(data.values())))
    actual_sequences = len({r["sequence"] for r in next(iter(data.values())).values()})
    if (actual_images, actual_sequences) != (expected_images, expected_sequences):
        raise ValueError(f"Expected {expected_images} images / {expected_sequences} sequences, "
                         f"found {actual_images} / {actual_sequences}")
    penalty = float(cfg["fp_penalty"])
    pairs = cfg["comparisons"]
    comparisons, rows = [], []
    for pair in pairs:
        m, r = pair["method"], pair["reference"]
        result, per_seq = analyze_pair(data, m, r, penalty, int(cfg["bootstrap_samples"]),
                                       int(cfg["seed"]))
        comparisons.append(result)
        rows.extend(per_seq)
        wins = result["sequence_wins_ties_losses"]
        print(f"{m} vs {r}:")
        print(f"  Δsmall TP={result['delta_small_tp']:+d}  ΔFP={result['delta_fp']:+d}  "
              f"Δutility={result['delta_utility_total']:+.2f} "
              f"({result['delta_utility_per_image']:+.4f}/image)")
        print(f"  Sequence-bootstrap exploratory 95% CI utility/image: "
              f"{result['sequence_bootstrap_95ci_per_image']} | "
              f"wins/ties/losses={wins['wins']}/{wins['ties']}/{wins['losses']}")
    args.out.mkdir(parents=True, exist_ok=False)
    payload = {
        "task": "Step 9H exploratory paired detection utility, not independent final test",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(source), "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fp_penalty": penalty, "bootstrap_samples": int(cfg["bootstrap_samples"]),
        "seed": int(cfg["seed"]), "images": actual_images, "sequences": actual_sequences,
        "methods": methods, "comparisons": comparisons,
        "limitations": [
            "Sequence clusters are inferred from dataset video IDs; effective sample size is small.",
            "Exploratory 95% percentile bootstrap CI, not confirmatory hypothesis testing.",
            "Same FLIR validation-domain development set has been repeatedly analyzed; not untouched final evaluation.",
            "Diagnostic small-object TP and FP at specified detector thresholds, not standardized mAP.",
            "FP penalty of 0.25 is a chosen utility weight; conclusions depend on this choice.",
            "Does not establish end-to-end latency savings or detector improvements.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (args.out / "sequence_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "reference", "sequence", "images",
                                                     "delta_tp", "delta_fp", "delta_utility", "delta_utility_per_image"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"SAVED — {args.out / 'summary.json'} and sequence_differences.csv")


if __name__ == "__main__":
    main()
