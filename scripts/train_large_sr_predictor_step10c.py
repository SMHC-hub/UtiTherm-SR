"""Step 10C — train V3 on Large-FSRCNN-specific ROI labels only.

Uses the original 4-head 31,876-parameter UtilityPredictor and the V2 trainer's
ranking loss, but no Bicubic labels or V2 pretrained weights. VAL selects the
checkpoint; separate TEST sequences are evaluated exactly once afterwards.

IMPORTANT: These are single-ROI counterfactuals on YOLO TRAIN frames, NOT an
independent detector test or an observed combined-4-ROI detection result.

Usage (from project root):
  python scripts/train_large_sr_predictor_step10c.py --smoke
  python scripts/train_large_sr_predictor_step10c.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from train_utility_predictor import load_measurements, ranking_report, weight_by_head
from train_utility_predictor_v2 import (ImageTilesDataset, head_metrics,
                                         run_epoch, sha256, split_v2,
                                         topk_per_image, load_v1, sequence_bootstrap_delta)
from utitherm.utility.predictor import HEADS, UtilityPredictor

REQUIRED = {
    "annotations_csv", "collection_manifest", "sr_checkpoint", "train_lr",
    "output", "canonical_checkpoint", "v2_checkpoint", "v2_source_csv",
    "v2_split_manifest", "expected_images", "expected_sequences",
    "seed", "val_fraction", "test_fraction", "input_width", "input_height",
    "batch_images", "workers", "epochs", "patience", "learning_rate",
    "weight_decay", "pos_weight_cap", "ranking_loss_weight",
    "ranking_logit_scale", "false_positive_penalty", "budget_tiles", "device",
}
PROTOCOL = "STEP_10A_LARGE_FSRCNN_H10_SINGLE_ROI_LABELS_V1"


def check_collection(rows: list[dict], collection: dict, config: dict,
                     csv_file: Path, sr_file: Path) -> dict:
    """Verify that labels come from the frozen 300-frame Large-FSRCNN collection."""
    if not isinstance(collection, dict) or collection.get("protocol") != PROTOCOL:
        raise ValueError("Step 10A Large FSRCNN label provenance required")
    if collection.get("intervention") != "large_fsrcnn_x2_roi_h10_linear_background_feather8":
        raise ValueError("ROI labels are not Large FSRCNN halo-10 interventions")
    if int(collection.get("halo", -1)) != 10 or int(collection.get("num_tiles", -1)) != 16:
        raise ValueError("Unexpected halo or region grid")
    if not sr_file.is_file() or collection.get("sr_checkpoint_sha256") != sha256(sr_file):
        raise ValueError("Large-FSRCNN checkpoint hash differs from label collection")
    names = {r["image"] for r in rows}
    sequences = {r["sequence"] for r in rows}
    frozen_names = collection.get("selected_images")
    if (not isinstance(frozen_names, list) or len(frozen_names) != len(set(frozen_names))
            or names != set(frozen_names)):
        raise ValueError("Label images do not match frozen collection manifest")
    nimages, nseqs = int(config["expected_images"]), int(config["expected_sequences"])
    if (len(names), len(sequences), len(rows)) != (nimages, nseqs, nimages * 16):
        raise ValueError("Expected complete 300-image, 33-sequence ROI dataset")
    if csv_file.name != "roi_utility.csv":
        raise ValueError("Expected the original collected roi_utility.csv, not a mixed label CSV")
    return {"images": len(names), "sequences": len(sequences), "regions": len(rows),
            "sr_checkpoint_sha256": sha256(sr_file), "roi_csv_sha256": sha256(csv_file)}


def split_large(rows: list[dict], seed: int, val_fraction: float, test_fraction: float):
    """Reuse tested V2 splitter with NO old/Bicubic intervention records."""
    training, validation, test, groups = split_v2([], rows, seed, val_fraction, test_fraction)
    if not (training and validation and test):
        raise ValueError("Empty sequence group")
    return training, validation, test, groups



def verify_v2_paired_split(v3_groups: dict, v2_split: dict,
                           v2_checkpoint: Path, v2_source_csv: Path,
                           large_rows: list[dict], bicubic_rows: list[dict]) -> dict:
    """Require the exact V2 new-sequence holdouts and matching 300-image keys.

    Old V1 sequences in V2 training are allowed; V3 never uses them.
    Checkpoint source hash must match Bicubic fresh ROI label CSV. V2's TEST
    was never used for V2 checkpoint selection, subject to later development
    decisions. This remains detector TRAIN-domain, NOT independent testing.
    """
    if not isinstance(v2_split, dict) or "sequences" not in v2_split:
        raise ValueError("V2 split manifest lacks sequences")
    vs = {k: set(v2_split["sequences"].get(k, [])) for k in ("train", "val", "test")}
    gs = {k: set(v3_groups[k]) for k in ("train", "val", "test")}
    if not gs["test"] == vs["test"] or not gs["val"] == vs["val"] or not gs["train"].issubset(vs["train"]):
        raise ValueError("V2 and V3 new-sequence train/validation/test splits differ; comparison not paired")
    if gs["test"] & vs["train"] or gs["test"] & vs["val"]:
        raise ValueError("V2 predictor saw V3 TEST sequences during training/validation")
    large_keys = {(r["image"], int(r["tile_index"])) for r in large_rows}
    bicubic_keys = {(r["image"], int(r["tile_index"])) for r in bicubic_rows}
    if large_keys != bicubic_keys:
        raise ValueError("Large and Bicubic fresh label records are not matched")
    if not v2_checkpoint.is_file() or not v2_source_csv.is_file():
        raise FileNotFoundError("V2 checkpoint/source CSV missing")
    state = torch.load(v2_checkpoint, map_location="cpu", weights_only=True)
    if (tuple(state.get("heads", [])) != tuple(HEADS)
            or list(state.get("input_size", [])) != [80, 64]
            or state.get("source_csv_sha256") != sha256(v2_source_csv)):
        raise ValueError("V2 checkpoint metadata/source provenance mismatch")
    return {"v2_checkpoint_sha256": sha256(v2_checkpoint),
            "v2_source_csv_sha256": sha256(v2_source_csv),
            "matched_test_sequences": len(gs["test"])}


def bootstrap_random_expectation(selections: dict, rows: list[dict],
                                 budget: int, seed: int, draws: int = 2000) -> dict:
    """Compare selected *marginal* utilities to exact random budget expectation.

    Sequence-level cluster resampling preserves frame correlation and uneven
    sequence lengths. Not a confidence interval for multi-ROI detector utility.
    """
    if not (1 <= budget <= 16 and draws >= 100 and selections):
        raise ValueError("Invalid bootstrap parameters")
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image"]].append(row)
    groups = defaultdict(list)
    for name, selection in selections.items():
        rec = by_image.get(name)
        if rec is None or len(rec) != 16:
            raise ValueError("Expected 16 regions per image")
        if selection["sequence"] != rec[0]["sequence"]:
            raise ValueError("Sequence mismatch")
        expected_random = budget / 16 * sum(float(r["score"]) for r in rec)
        groups[selection["sequence"]].append(selection["utility"] - expected_random)
    if len(groups) < 2:
        raise ValueError("Need multiple independent sequences")
    keys = sorted(groups)
    by_group = [np.array(groups[k], dtype=np.float64) for k in keys]
    observed = float(sum(a.sum() for a in by_group) / sum(len(a) for a in by_group))
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        picks = rng.integers(0, len(keys), size=len(keys))
        result[i] = sum(by_group[j].sum() for j in picks) / sum(len(by_group[j]) for j in picks)
    ci = np.quantile(result, [.025, .975])
    return {"delta_predicted_minus_expected_random_per_image": round(observed, 6),
            "sequence_bootstrap_exploratory_95ci": [round(float(x), 6) for x in ci],
            "num_sequences": len(keys),
            "note": "SINGLE-ROI marginal-utility proxy only; not combined-ROI detection benefit."}


def train(args: argparse.Namespace) -> Path:
    cfg_file = args.config
    if not cfg_file.is_file():
        raise FileNotFoundError(cfg_file)
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or REQUIRED - set(cfg):
        raise ValueError(f"Missing config keys: {sorted(REQUIRED - set(cfg if isinstance(cfg, dict) else {}))}")
    if abs(float(cfg["false_positive_penalty"]) - .25) > 1e-9:
        raise ValueError("Utility requires frozen 0.25 false-positive penalty")
    if (int(cfg["input_width"]), int(cfg["input_height"])) != (80, 64):
        raise ValueError("Require V1/V2-compatible predictor input encoding (80,64)")
    if int(cfg["budget_tiles"]) != 4:
        raise ValueError("Require frozen selection budget of four tiles")
    if not (1 <= int(cfg["batch_images"]) <= 8 and int(cfg["workers"]) >= 0):
        raise ValueError("Unsafe batch/worker configuration")
    if not (int(cfg["epochs"]) >= 1 and int(cfg["patience"]) >= 1
            and float(cfg["learning_rate"]) > 0 and float(cfg["weight_decay"]) >= 0
            and float(cfg["ranking_loss_weight"]) >= 0
            and float(cfg["ranking_logit_scale"]) > 0):
        raise ValueError("Invalid training hyperparameters")
    out = args.out or Path(str(cfg["output"]) + ("_smoke" if args.smoke else ""))
    canonical = Path(cfg["canonical_checkpoint"])
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {out}")
    if not args.smoke and canonical.exists():
        raise FileExistsError(f"Refusing to overwrite canonical V3 checkpoint: {canonical}")
    img_root = Path(cfg["train_lr"])
    if (img_root.name != "train" or img_root.parent.name != "images"
            or "yolo_flir_lr_x2" not in img_root.parts or not img_root.is_dir()):
        raise ValueError("Only detector-TRAIN LR images are allowed")
    csv_file = Path(cfg["annotations_csv"])
    collection_path = Path(cfg["collection_manifest"])
    sr_file = Path(cfg["sr_checkpoint"])
    rows = load_measurements(csv_file, float(cfg["false_positive_penalty"]))
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    provenance = check_collection(rows, collection, cfg, csv_file, sr_file)
    seed = int(cfg["seed"])
    if args.smoke:
        # Use a full sequence-only subset; never use partial 16-tile image rows.
        keep = set(sorted({r["sequence"] for r in rows})[:12])
        rows = [r for r in rows if r["sequence"] in keep]
    training, validation, test, groups = split_large(
        rows, seed, float(cfg["val_fraction"]), float(cfg["test_fraction"]))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    request = str(cfg["device"]).lower()
    if request not in ("cpu", "auto", "cuda", "0"):
        raise ValueError("device must be cpu, auto, cuda or 0")
    if request in ("cuda", "0") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    cuda = request in ("auto", "cuda", "0") and torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    if cuda:
        torch.backends.cudnn.benchmark = False
    v2_audit = None
    if not args.smoke:
        v2_path, v2_csv = Path(cfg["v2_checkpoint"]), Path(cfg["v2_source_csv"])
        bicubic_rows = load_measurements(v2_csv, .25)
        v2_meta = json.loads(Path(cfg["v2_split_manifest"]).read_text(encoding="utf-8"))
        v2_audit = verify_v2_paired_split(groups, v2_meta, v2_path, v2_csv, rows, bicubic_rows)
        print(f"V2 matched TEST sequence audit PASSED: {v2_audit['matched_test_sequences']} sequences", flush=True)
    ds = {name: ImageTilesDataset(items, img_root, (80,64)) for name, items in
          (("train", training), ("val", validation), ("test", test))}
    if min(len(v) for v in ds.values()) < 2:
        raise ValueError("Too few images in one sequence-separated split")
    loaders = {name: DataLoader(dataset, batch_size=int(cfg["batch_images"]),
                                shuffle=(name == "train"), num_workers=int(cfg["workers"]),
                                pin_memory=cuda) for name, dataset in ds.items()}
    model = UtilityPredictor().to(device)   # Intentional scratch initialization.
    weights = weight_by_head(training, float(cfg["pos_weight_cap"])).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    print(f"V3 {device} | images train/val/test={len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])} "
          f"| sequences={len(groups['train'])}/{len(groups['val'])}/{len(groups['test'])} "
          f"| parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    (out / "split_manifest.json").write_text(json.dumps({
        "seed": seed, "sequences": groups,
        "images": {name: len(dataset) for name, dataset in ds.items()},
        "source": str(csv_file), "source_sha256": provenance["roi_csv_sha256"],
        "collection_manifest_sha256": sha256(collection_path),
        "sr_checkpoint_sha256": provenance["sr_checkpoint_sha256"],
        "config_sha256": sha256(cfg_file),
        "notes": "No Bicubic rows or V2 pretraining; grouped split by sequence. Test not used for checkpoint selection."
    }, indent=2) + "\n", encoding="utf-8")
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, 2 if args.smoke else int(cfg["epochs"]) + 1):
        tr, _, _ = run_epoch(model, loaders["train"], device, criterion,
                             .25, float(cfg["ranking_loss_weight"]),
                             float(cfg["ranking_logit_scale"]), optimizer)
        va, _, _ = run_epoch(model, loaders["val"], device, criterion,
                             .25, float(cfg["ranking_loss_weight"]),
                             float(cfg["ranking_logit_scale"]))
        if not np.isfinite(va["total"]) or not np.isfinite(tr["total"]):
            raise RuntimeError("Nonfinite training/validation loss")
        improved = va["total"] < best - 1e-6
        if improved:
            best, stale = va["total"], 0
            torch.save({
                "model": model.state_dict(), "heads": list(HEADS),
                "input_size": [80, 64], "seed": seed, "epoch": epoch,
                "source_csv_sha256": provenance["roi_csv_sha256"],
                "utility_protocol": PROTOCOL,
                "sr_checkpoint_sha256": provenance["sr_checkpoint_sha256"],
                "halo": 10,
                "comment": "Step 10C SR-specific V3; detector-TRAIN labels; NO independent detector test"
            }, out / "best.pt")
        else:
            stale += 1
        history.append({"epoch": epoch, "train_total": tr["total"], "val_total": va["total"],
                        "train_bce": tr["bce"], "val_bce": va["bce"],
                        "train_ranking": tr["ranking"], "val_ranking": va["ranking"]})
        print(f"epoch {epoch:02}/{1 if args.smoke else int(cfg['epochs'])}: "
              f"train={tr['total']:.4f} val={va['total']:.4f} "
              f"(BCE={va['bce']:.4f}, rank={va['ranking']:.4f}) "
              f"best={best:.4f}", flush=True)
        if stale >= int(cfg["patience"]):
            print("Early stopping — validation combined loss did not improve", flush=True)
            break
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    # DO NOT use TEST to select the epoch or change hyperparameters.
    test_loss, predictions, y = run_epoch(model, loaders["test"], device, criterion,
                                           .25, float(cfg["ranking_loss_weight"]),
                                           float(cfg["ranking_logit_scale"]))
    records = ds["test"].flat_rows()
    proxy = ranking_report(records, predictions, 4, seed, .25)
    selections = topk_per_image(records, predictions, 4, .25)
    delta = bootstrap_random_expectation(selections, records, 4, seed)
    v2_comparison = None
    if not args.smoke:
        # V2 learned Bicubic targets, never Large SR; test it on the SAME V3 TEST
        # images only after V3 checkpoint has been selected on VAL.
        model_v2 = load_v1(Path(cfg["v2_checkpoint"]), device, (80,64))
        v2_loss, p2, _ = run_epoch(model_v2, loaders["test"], device, criterion,
                                   .25, float(cfg["ranking_loss_weight"]),
                                   float(cfg["ranking_logit_scale"]))
        selections_v2 = topk_per_image(records, p2, 4, .25)
        paired_delta = sequence_bootstrap_delta(selections_v2, selections, seed)
        v2_comparison = {
            "v2_provenance": v2_audit,
            "v2_test_loss_using_v3_label_targets": v2_loss,
            "v2_test_head_ap_large_sr_labels": head_metrics(y, p2),
            "v2_large_sr_ranking_proxy": ranking_report(records, p2, 4, seed, .25),
            "exploratory_paired_v3_minus_v2": {
                "delta_per_sequence_mean": paired_delta["delta_v2_minus_v1_per_sequence_mean"],
                "sequence_bootstrap_exploratory_95ci": paired_delta["sequence_bootstrap_exploratory_95ci"],
                "num_sequences": paired_delta["num_sequences"],
                "note": "V3 minus V2 on same V3 TEST groups, both pretrained on other sequences."
            }
        }
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    with (out / "test_selections.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "sequence", "tiles", "marginal_utility",
                                                  "selected_small_gains", "selected_small_losses",
                                                  "selected_fp_reductions", "selected_fp_increases"])
        writer.writeheader()
        for name in sorted(selections):
            p = selections[name]
            writer.writerow({"image": name, "sequence": p["sequence"],
                             "tiles": ",".join(map(str,p["tiles"])), "marginal_utility": p["utility"],
                             "selected_small_gains": p["small_gains"],
                             "selected_small_losses": p["small_losses"],
                             "selected_fp_reductions": p["fp_reductions"],
                             "selected_fp_increases": p["fp_increases"]})
    summary = {
        "task": "Step 10C — Large FSRCNN utility predictor V3 (training-domain labels)",
        "utc": datetime.now(timezone.utc).isoformat(),
        "training_target": "Large FSRCNN x2, halo 10, one region, fixed detector, fp_penalty=0.25",
        "checkpoint": str(out / "best.pt"), "epoch": state["epoch"],
        "best_validation_combined_loss": best, "device": str(device),
        "train_val_test_sequences": {k: len(v) for k, v in groups.items()},
        "train_val_test_images": {k: len(v) for k, v in ds.items()},
        "source": provenance, "matched_v2_comparison": v2_comparison,
        "test": {"loss": test_loss,
                                         "head_ap": head_metrics(y, predictions),
                                         "ranking_proxy": proxy,
                                         "vs_exact_random_expectation": delta},
        "limitations": [
            "Detector learned from these frames; no detector out-of-fold labels.",
            "V2 and V3 train on different targets; their matching V2/V3 TEST sequence split is verified against V2 saved manifest.",
            "These original V2 TEST sequences have already influenced later project development decisions; comparison is exploratory, not untouched final evaluation.",
            "TEST is only utility-model sequence-held-out; not an external detection test.",
            "Marginal region utilities summed as ranking proxies, NOT actual combined four-ROI outcomes.",
            "Do not repeatedly tune on TEST or repeatedly used FLIR development validation.",
            "Actual end-to-end detection utility, timing, multi-seed evaluation and independent final test remain necessary."
        ]
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.smoke:
        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "best.pt", canonical)
        print(f"Canonical V3 checkpoint: {canonical}", flush=True)
    print(f"DONE — {out / 'summary.json'} | best epoch={state['epoch']} "
          f"| TEST top-4 marginal proxy V3="
          f"{proxy['mean_marginal_utility_per_image']['predicted']:.3f} "
          f"random={proxy['mean_marginal_utility_per_image']['random']:.3f} "
          "(NOT observed combined-region detection improvement)", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   default=Path("configs/large_sr_predictor_step10c.yaml"))
    p.add_argument("--smoke", action="store_true", help="1 epoch; separate noncanonical output")
    p.add_argument("--out", type=Path, default=None, help="New output location; existing folders refused")
    train(p.parse_args())


if __name__ == "__main__":
    main()
