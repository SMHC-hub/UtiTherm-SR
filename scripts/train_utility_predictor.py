"""Step 5B: train a tile-utility ranking prototype (no GT inputs to model).

Docker commands:
  python scripts/train_utility_predictor.py --config configs/utility_predictor.yaml --smoke
  python scripts/train_utility_predictor.py --config configs/utility_predictor.yaml

Scientific scope: labels are collected on images seen by the fixed YOLO detector.
Only sequence-separated internal validation is possible here; this is NOT an
independent measurement of detection gains, novelty, or compute savings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utitherm.utility.predictor import HEADS, UtilityPredictor, encode_tile, outcome_labels, ranking_score

REQUIRED = {"annotations_csv", "train_lr", "output", "canonical_checkpoint", "seed", "val_fraction",
            "input_width", "input_height", "batch", "workers", "epochs", "patience",
            "learning_rate", "weight_decay", "pos_weight_cap", "budget_tiles", "device", "false_positive_penalty"}
CSV_FIELDS = {"image", "sequence", "tile_index", "tile_x0_hr", "tile_y0_hr", "tile_x1_hr",
              "tile_y1_hr", "delta_small_tp", "delta_fp", "utility_score", "positive_label"}


def load_measurements(path: Path, fp_penalty: float = 0.25) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 5A measurements: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or CSV_FIELDS - set(reader.fieldnames):
            raise ValueError(f"Missing CSV columns: {sorted(CSV_FIELDS - set(reader.fieldnames or []))}")
        rows = []
        unique = set()
        for row in reader:
            if not row["image"].lower().endswith(".png") or Path(row["image"]).name != row["image"] or "\\" in row["image"]:
                raise ValueError("Expected safe basename PNG image names")
            if not row["sequence"] or row["sequence"] != row["image"].split("-frame-", 1)[0]:
                raise ValueError(f"Sequence/image mismatch: {row['image']}")
            tile = int(row["tile_index"])
            key = (row["image"], tile)
            if tile not in range(16) or key in unique:
                raise ValueError(f"Duplicate or invalid ROI record: {key}")
            unique.add(key)
            rect = tuple(int(row[x]) for x in ("tile_x0_hr", "tile_y0_hr", "tile_x1_hr", "tile_y1_hr"))
            expected_rect = ((tile % 4) * 160, (tile // 4) * 128, (tile % 4 + 1) * 160, (tile // 4 + 1) * 128)
            if rect != expected_rect:
                raise ValueError(f"Invalid ROI coordinates: {key}")
            # Strictly enforce target consistency to avoid silently learning bad labels.
            score = float(row["utility_score"])
            expected = int(row["delta_small_tp"]) - fp_penalty * int(row["delta_fp"])
            if not np.isfinite(score) or abs(score - expected) > 1e-4 or int(row["positive_label"]) != int(score > 0):
                raise ValueError(f"ROI utility-label mismatch: {key}")
            row["rect"] = rect
            row["target"] = outcome_labels(row)
            row["score"] = score
            rows.append(row)
    if not rows:
        raise ValueError("Utility measurement CSV is empty")
    by_image = Counter(r["image"] for r in rows)
    bad = [k for k, n in by_image.items() if n != 16]
    if bad:
        raise ValueError(f"Expected all 16 tiles per image; incomplete: {bad[:3]}")
    return rows


def grouped_split(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split entire sequences, never individual tile records."""
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")
    sequences = sorted({r["sequence"] for r in rows})
    if len(sequences) < 4:
        raise ValueError("Need at least four distinct sequences for a meaningful split")
    random.Random(seed).shuffle(sequences)
    nval = min(len(sequences)-1, max(1, round(len(sequences) * val_fraction)))
    val_sequences = set(sequences[:nval])
    training = [r for r in rows if r["sequence"] not in val_sequences]
    validation = [r for r in rows if r["sequence"] in val_sequences]
    assert {r["sequence"] for r in training}.isdisjoint({r["sequence"] for r in validation})
    return training, validation


class UtilityDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path, size: tuple[int, int]):
        self.rows, self.root, self.size = rows, root, size
        for fname in {row["image"] for row in rows}:
            if not (root / fname).is_file():
                raise FileNotFoundError(f"Training LR image missing: {root / fname}")

    @lru_cache(maxsize=128)
    def _frame(self, image_name: str) -> np.ndarray:
        img = cv2.imread(str(self.root / image_name), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (256, 320):
            raise ValueError(f"Expected 320x256 training LR: {image_name}")
        return img

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        item = self.rows[index]
        x, coord = encode_tile(self._frame(item["image"]), item["rect"], input_size=self.size)
        return x, coord, torch.tensor(item["target"], dtype=torch.float32)


def weight_by_head(training: list[dict], cap: float) -> torch.Tensor:
    if cap < 1:
        raise ValueError("pos_weight_cap must be >= 1")
    labels = np.stack([r["target"] for r in training])
    pos = labels.sum(0)
    neg = len(labels) - pos
    weights = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.from_numpy(np.clip(weights, 1, cap).astype(np.float32))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    losses, logits_all, y_all = [], [], []
    with torch.inference_mode():
        for x, coords, y in loader:
            x, coords, y = x.to(device), coords.to(device), y.to(device)
            logits = model(x, coords)
            losses.append((float(criterion(logits, y)), len(y)))
            logits_all.append(torch.sigmoid(logits).cpu().numpy())
            y_all.append(y.cpu().numpy())
    loss = sum(v * n for v, n in losses) / sum(n for _, n in losses)
    return loss, np.concatenate(logits_all), np.concatenate(y_all)


def ranking_report(rows: list[dict], predictions: np.ndarray, budget_tiles: int, seed: int, fp_penalty: float = 0.25) -> dict:
    """Marginal-utility proxy only: single-tile effects are not additive in general."""
    if len(rows) != len(predictions) or budget_tiles < 1:
        raise ValueError("Invalid rankings or budget")
    scores = ranking_score(predictions, fp_penalty=fp_penalty)
    by_image = defaultdict(list)
    for i, r in enumerate(rows):
        by_image[r["image"]].append(i)
    results = defaultdict(list)
    selected_records = []
    for image in sorted(by_image):
        ids = by_image[image]
        if len(ids) != 16:
            raise ValueError("Must have all 16 regions for each image")
        n = min(budget_tiles, len(ids))
        ordered = sorted(ids, key=lambda i: (-float(scores[i]), int(rows[i]["tile_index"])))
        actual = sorted(ids, key=lambda i: (-float(rows[i]["score"]), int(rows[i]["tile_index"])))
        # Fair random controls: select exactly n tiles, as does the predictor.
        rng = random.Random(f"{seed}:{image}")
        random_total = float(np.mean([
            sum(rows[i]["score"] for i in rng.sample(ids, n)) for _ in range(100)
        ]))
        results["predicted"].append(sum(rows[i]["score"] for i in ordered[:n]))
        results["random"].append(random_total)
        results["oracle"].append(sum(rows[i]["score"] for i in actual[:n]))
        results["first_four"].append(sum(rows[i]["score"] for i in sorted(ids, key=lambda i: int(rows[i]["tile_index"]))[:n]))
        selected_records.extend(ordered[:n])
    return {"budget_tiles": budget_tiles, "num_images": len(by_image),
            "mean_marginal_utility_per_image": {k: round(float(np.mean(v)), 6) for k, v in results.items()},
            "selected_small_gains": int(sum(rows[i]["target"][0] > 0 for i in selected_records)),
            "selected_fp_reductions": int(sum(rows[i]["target"][1] > 0 for i in selected_records)),
            "selected_harmful_small_losses": int(sum(rows[i]["target"][2] > 0 for i in selected_records)),
            "note": "Single-region counterfactual utilities summed as a RANKING PROXY only; multi-ROI combined detection gains have NOT been evaluated."}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/utility_predictor.yaml"))
    ap.add_argument("--smoke", action="store_true", help="One epoch on up to 40 sequences; no canonical checkpoint")
    ap.add_argument("--out", type=Path, help="Override output directory (never overwrite)")
    args = ap.parse_args()
    if not args.config.is_file():
        ap.error(f"Config file missing: {args.config}")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or REQUIRED - set(cfg):
        ap.error(f"Missing configuration keys: {sorted(REQUIRED - set(cfg if isinstance(cfg,dict) else {}))}")
    output = args.out or Path(str(cfg["output"]) + ("_smoke" if args.smoke else ""))
    if output.exists():
        ap.error(f"Experiment directory already exists: {output}. Select a fresh --out")
    source = Path(cfg["train_lr"])
    if source.name != "train" or source.parent.name != "images" or "yolo_flir_lr_x2" not in source.parts:
        ap.error("For leakage prevention train_lr MUST point to data/processed/yolo_flir_lr_x2/images/train")
    if not source.is_dir():
        ap.error(f"Missing training images: {source}")
    rows = load_measurements(Path(cfg["annotations_csv"]), float(cfg["false_positive_penalty"]))
    if args.smoke:
        # Sample full sequences, never a mixture of tiles, for quick pipeline validation.
        all_seqs = sorted({r["sequence"] for r in rows})
        rng = random.Random(int(cfg["seed"]))
        rng.shuffle(all_seqs)
        selected = set(all_seqs[:40])
        rows = [r for r in rows if r["sequence"] in selected]
    train_rows, val_rows = grouped_split(rows, float(cfg["val_fraction"]), int(cfg["seed"]))
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    use_gpu = (str(cfg["device"]).lower() in ("auto", "cuda", "0") and torch.cuda.is_available())
    if str(cfg["device"]).lower() in ("cuda", "0") and not use_gpu:
        ap.error("CUDA requested but not found")
    device = torch.device("cuda:0" if use_gpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
    size = (int(cfg["input_width"]), int(cfg["input_height"]))
    if min(size) < 16:
        ap.error("input size must be >=16")
    train_ds, val_ds = UtilityDataset(train_rows, source, size), UtilityDataset(val_rows, source, size)
    batch_size = int(cfg["batch"])
    if batch_size < 1:
        ap.error("batch must be positive")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=int(cfg["workers"]), pin_memory=use_gpu)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=int(cfg["workers"]), pin_memory=use_gpu)
    model = UtilityPredictor().to(device)
    weight = weight_by_head(train_rows, float(cfg["pos_weight_cap"])).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                              weight_decay=float(cfg["weight_decay"]))
    epochs = 1 if args.smoke else int(cfg["epochs"])
    if epochs < 1:
        ap.error("epochs must be positive")
    output.mkdir(parents=True)
    (output / "split_manifest.json").write_text(json.dumps({
        "seed": seed, "train_sequences": sorted({r["sequence"] for r in train_rows}),
        "val_sequences": sorted({r["sequence"] for r in val_rows}),
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "train_head_positive_counts": dict(zip(HEADS, map(int, np.stack([r["target"] for r in train_rows]).sum(0)))),
        "val_head_positive_counts": dict(zip(HEADS, map(int, np.stack([r["target"] for r in val_rows]).sum(0)))),
        "intervention_csv_sha256": hashlib.sha256(Path(cfg["annotations_csv"]).read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Device: {device} | train/validation sequences: "
          f"{len({r['sequence'] for r in train_rows})}/{len({r['sequence'] for r in val_rows})} | "
          f"rows: {len(train_rows)}/{len(val_rows)} | parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Class positive weights: {dict(zip(HEADS, [round(float(w), 2) for w in weight.cpu()]))}", flush=True)
    history = []
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        sumloss = 0.0
        count = 0
        for x, coords, y in train_loader:
            x, coords, y = x.to(device, non_blocking=True), coords.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(x, coords)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            sumloss += float(loss.detach()) * len(y)
            count += len(y)
        val_loss, probs, true = evaluate(model, val_loader, device, criterion)
        train_loss = sumloss / max(count, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        improved = val_loss < best_loss - 1e-6
        if improved:
            best_loss, stale = val_loss, 0
            torch.save({"model": model.state_dict(), "heads": list(HEADS),
                        "input_size": list(size), "seed": seed, "epoch": epoch,
                        "source_csv_sha256": hashlib.sha256(Path(cfg["annotations_csv"]).read_bytes()).hexdigest(),
                        "comment": "Step 5B exploratory, not independent detection evaluation"}, output / "best.pt")
        else:
            stale += 1
        print(f"epoch {epoch:02}/{epochs}: train BCE={train_loss:.4f}, val BCE={val_loss:.4f}, best={best_loss:.4f}", flush=True)
        if stale >= int(cfg["patience"]):
            print("Early stopping (validation BCE did not improve)")
            break
    # Reload BEST checkpoint before producing final predictions, even after early stopping.
    saved = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    _, preds, labels = evaluate(model, val_loader, device, criterion)
    metrics = {}
    for j, head in enumerate(HEADS):
        positives = int(labels[:, j].sum())
        metrics[head] = {"positive_count": positives,
                         "positive_rate": float(labels[:, j].mean()),
                         "average_precision": float(average_precision_score(labels[:, j], preds[:, j])) if 0 < positives < len(labels) else None,
                         "note": "AP undefined without both classes" if positives in (0, len(labels)) else ""}
    ranking = ranking_report(val_rows, preds, int(cfg["budget_tiles"]), seed, float(cfg["false_positive_penalty"]))
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    with (output / "val_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["image", "sequence", "tile_index", "true_utility", "predicted_ranking_score"]
        cols += [f"true_{h}" for h in HEADS] + [f"pred_prob_{h}" for h in HEADS]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        rank = ranking_score(preds, fp_penalty=float(cfg["false_positive_penalty"]))
        for i, row in enumerate(val_rows):
            item = {"image": row["image"], "sequence": row["sequence"],
                    "tile_index": int(row["tile_index"]), "true_utility": row["score"],
                    "predicted_ranking_score": float(rank[i])}
            item.update({f"true_{h}": int(row["target"][j]) for j, h in enumerate(HEADS)})
            item.update({f"pred_prob_{h}": float(preds[i, j]) for j, h in enumerate(HEADS)})
            writer.writerow(item)
    summary = {"task": "UtiTherm-SR Step 5B exploratory tile utility predictor",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "device": str(device), "checkpoint": str(output / "best.pt"), "best_epoch": saved["epoch"],
               "best_val_loss": best_loss, "train_rows": len(train_rows), "val_rows": len(val_rows),
               "val_head_metrics": metrics, "budget_ranking_proxy": ranking,
               "notes": ["Ground truth only supplies OFFLINE training targets; inputs use grayscale LR tile, LR whole-frame context and geometric tile coordinates.",
                         "This is a TRAIN-domain sequence-split evaluation, not held-out detector validation.",
                         "Fixed YOLO detector was trained on these training frames; label optimism and correlated scenes are unresolved.",
                         "Head scores estimate event probabilities, not expected utility magnitudes.",
                         "Summing individual tile effects does not measure combined multi-tile enhancement or wall-clock compute savings.",
                         "Before paper: out-of-fold ROI labels, more sequences, fair held-out test, calibration/PR analyses, compare random and confidence ROIs end-to-end."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.smoke:
        canonical = Path(cfg["canonical_checkpoint"])
        canonical.parent.mkdir(parents=True, exist_ok=True)
        if canonical.exists():
            print(f"WARNING: canonical checkpoint already exists ({canonical}); left unchanged. Save experiment checkpoint separately.")
        else:
            shutil.copy2(output / "best.pt", canonical)
            print(f"Canonical checkpoint: {canonical}")
    print(f"DONE — {output/'summary.json'} | predictor top-{ranking['budget_tiles']} marginal utility "
          f"{ranking['mean_marginal_utility_per_image']['predicted']:.3f} vs random "
          f"{ranking['mean_marginal_utility_per_image']['random']:.3f}; this is a PROXY, not observed end-to-end gain.", flush=True)


if __name__ == "__main__":
    main()
